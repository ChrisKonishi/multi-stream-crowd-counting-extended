import os
import os.path as osp
import torch
import numpy as np
import sys
import pickle
import torch.nn as nn

from architecture.crowd_count import CrowdCounter
from architecture import network
from architecture.data_loader import ImageDataLoader
from architecture.timer import Timer
from architecture import utils
from architecture.evaluate_model import evaluate_model
from architecture.LossPlotter import LossPlotter

import argparse

from manage_data import dataset_loader
from manage_data.utils import Logger, mkdir_if_missing
from architecture.network import np_to_variable

def train_gan(train_test_unit, out_dir_root, args):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    output_dir_model = osp.join(output_dir, 'models')
    mkdir_if_missing(output_dir_model)
    if args.resume:
        sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'), mode='a')
        plotter = LossPlotter(out_dir_root, mode='a')
    else:
        sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'))
        plotter = LossPlotter(out_dir_root, mode='w')
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
    lrc = args.lrc

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
        network.weights_normal_init(net.net, dev=0.01)

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

    #optmizers and loss
    optimizerG = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.net.parameters()), lr=lr)
    optimizerD = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.gan_net.parameters()), lr=lrc)

    mse_criterion = nn.MSELoss()

    # training
    train_lossG = 0
    train_lossD = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()

    # gan labels
    real_label = 1
    fake_label = 0

    netD = net.gan_net
    netG = net.net

    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, batch_size = args.train_batch)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size = 1)
    

    for epoch in range(start_step, end_step+1):
        step = 0
        train_lossG = 0
        train_lossD = 0
        train_lossG_mse = 0
        train_lossG_gan = 0


        for blob in data_loader:
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            step = step + args.train_batch
            im_data = blob['data']
            gt_data = blob['gt_density']
            im_data_norm = im_data / 127.5 - 1. #normalize between -1 and 1
            gt_data *= args.den_scale_factor

            im_data_norm = network.np_to_variable(im_data_norm, is_cuda=True, is_training=True)
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=True)

            errD_epoch = 0

            for critic_epoch in range(args.ncritic):
                netD.zero_grad()
                netG.zero_grad()

                #real data discriminator
                b_size = gt_data.size(0)
                output_real = netD(gt_data).view(-1)

                #fake data discriminator
                density_map = netG(im_data_norm)
                output_fake = netD(density_map.detach()).view(-1)

                errD = -(torch.mean(output_real) - torch.mean(output_fake))
                errD.backward()
                optimizerD.step()

                for p in netD.parameters():
                    p.data.clamp_(-0.01, 0.01)

                errD_epoch += errD.data.item()

            errD_epoch /= args.ncritic

            #Generator update
            netG.zero_grad()
            output_fake = netD(density_map).view(-1)
            errG_gan = -torch.mean(output_fake)
            errG_mse = mse_criterion(density_map, gt_data)
            #errG = (1-args.alpha)*errG_mse + args.alpha*errG_gan
            errG = errG_mse + args.alpha*errG_gan
            errG.backward()
            optimizerG.step()

            train_lossG += errG.data.item()
            train_lossG_mse += errG_mse.data.item()
            train_lossG_gan += errG_gan.data.item()
            train_lossD += errD_epoch
            density_map = density_map.data.cpu().numpy()
            density_map/=args.den_scale_factor
            gt_data = gt_data.data.cpu().numpy()
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

        plotter.report(train_lossG_mse, train_lossG_gan, train_lossD)
        plotter.save()
        plotter.plot()

        print("Epoch: {0}, MAE: {1:.4f}, MSE: {2:.4f}, lossG: {3:.4f}, lossG_mse: {4:.4f}, lossG_gan: {5:.4f}, lossD: {6:.4f}".format(epoch, mae, mse, train_lossG, train_lossG_mse, train_lossG_gan, train_lossD))
        print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}".format(best_mae, best_mse, best_model))
        print("Patience: {0}/{1}".format(current_patience, args.patience))
        sys.stdout.close_open()

        if current_patience > args.patience and args.patience > -1:
            break