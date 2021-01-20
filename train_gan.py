import os
import os.path as osp
import torch
import numpy as np
import sys
import pickle

from architecture.crowd_count import CrowdCounter
from architecture import network
from architecture.data_loader import ImageDataLoader
from architecture.timer import Timer
from architecture import utils
from architecture.evaluate_model import evaluate_model

import argparse

from manage_data import dataset_loader
from manage_data.utils import Logger, mkdir_if_missing

def train_gan(train_test_unit, out_dir_root, args):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    output_dir_model = osp.join(output_dir, 'models')
    mkdir_if_missing(output_dir_model)
    if args.resume:
        sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'), mode='a')
    else:
        sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset_name = train_test_unit.metadata['name']
    train_path = train_test_unit.train_dir_img
    train_gt_path = train_test_unit.train_dir_den
    val_path =train_test_unit.val_dir_img
    val_gt_path = train_test_unit.val_dir_den

    #training configuration
    start_step = args.start_epoch
    end_step = args.max_epoch
    lr = args.lr

    #log frequency
    disp_interval = args.train_batch*20

    # ------------
    rand_seed = args.seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

    best_mae = sys.maxsize # best mae
    current_patience = 0

    # load net
    net = CrowdCounter(model = args.model)
    if not args.resume :
        network.weights_normal_init(net, dev=0.01)

    else:
        if args.resume[-3:] == '.h5': #don't use this option!
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            if args.last_model:
                pretrained_model = osp.join(resume_dir, 'last_model.h5')
                f = open(osp.join(resume_dir, "current_values.bin"), "rb")
                current_patience = pickle.load(f)
                f.close()
            else:
                pretrained_model = osp.join(resume_dir, 'best_model.h5')
                current_patience = 0
            f = open(osp.join(resume_dir, "best_values.bin"), "rb")
            best_mae, best_mse, best_model, _ = pickle.load(f)
            f.close()
            print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}, Current patience: {3}".format(best_mae, best_mse, best_model, current_patience))
                
        network.load_net(pretrained_model, net)
        print('Will apply fine tunning over', pretrained_model)
    net.cuda()
    net.train()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    # training
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()

    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, batch_size = args.train_batch)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size = 1)
    

    for epoch in range(start_step, end_step+1):
        step = 0
        train_loss = 0
        for blob in data_loader:
            optimizer.zero_grad()
            step = step + args.train_batch
            im_data = blob['data']
            gt_data = blob['gt_density']
            im_data_norm = im_data / 127.5 - 1. #normalize between -1 and 1
            gt_data *= args.den_scale_factor

            density_map = net(im_data_norm, gt_data = gt_data)
            loss = net.loss

            
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            density_map = density_map.data.cpu().numpy()
            density_map/=args.den_scale_factor
            gt_data/=args.den_scale_factor

            step_cnt += 1
            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration
                train_batch_size = gt_data.shape[0]
                gt_count = np.sum(gt_data.reshape(train_batch_size, -1), axis = 1)
                et_count = np.sum(density_map.reshape(train_batch_size, -1), axis = 1)

                print("epoch: {0}, step {1}/{5}, Time: {2:.4f}s, gt_cnt[0]: {3:.4f}, et_cnt[0]: {4:.4f}, mean_diff: {6:.4f}".format(epoch, step, 1./fps, gt_count[0],et_count[0], data_loader.num_samples, np.mean(np.abs(gt_count - et_count))))
                re_cnt = True

            if re_cnt:
                t.tic()
                re_cnt = False

        save_name = os.path.join(output_dir_model, '{}_{}_{}.h5'.format(train_test_unit.to_string(), dataset_name,epoch))
        network.save_net(save_name, net)
        network.save_net(os.path.join(output_dir, "last_model.h5"), net)

        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val, model = args.model, save_test_results=args.save_plots, plot_save_dir=osp.join(output_dir, 'plot-results-train/'), den_scale_factor = args.den_scale_factor)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            current_patience = 0
            best_model = '{}_{}_{}.h5'.format(train_test_unit.to_string(),dataset_name,epoch)
            network.save_net(os.path.join(output_dir, "best_model.h5"), net)
            f = open(os.path.join(output_dir, "best_values.bin"), "wb")
            pickle.dump((best_mae, best_mse, best_model, current_patience), f)
            f.close()

        else:
            current_patience += 1

        f = open(os.path.join(output_dir, "current_values.bin"), "wb")
        pickle.dump(current_patience, f)
        f.close()

        print("Epoch: {0}, MAE: {1:.4f}, MSE: {2:.4f}, loss: {3:.4f}".format(epoch, mae, mse, train_loss))
        print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}".format(best_mae, best_mse, best_model))
        print("Patience: {0}/{1}".format(current_patience, args.patience))
        sys.stdout.close_open()

        if current_patience > args.patience and args.patience > -1:
            break