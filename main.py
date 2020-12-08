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

parser = argparse.ArgumentParser(description='Multi-stream crowd counting')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='ucf-cc-50',
                    choices=dataset_loader.get_names(),  help="dasaset 'ucf-cc-50', 'shanghai-tech', 'jhu_crowd'. Default, 'ucf-cc-50'")
#Data augmentation hyperpameters
parser.add_argument('--force-den-maps', action='store_true', help="force generation of density maps for original dataset, by default it is generated only once")
parser.add_argument('--force-augment', action='store_true', help="force generation of augmented data, by default it is generated only once")
parser.add_argument('--displace', default=70, type=int,help="displacement for sliding window in data augmentation, default 70")
parser.add_argument('--size-x', default=256, type=int, help="width of sliding window in data augmentation, default 300")
parser.add_argument('--size-y', default=256, type=int, help="height of sliding window in data augmentation, default 200")
parser.add_argument('--people-thr', default=0, type=int, help="minimum quantitie of people in each sliding window in data augmentation, default 0")
parser.add_argument('--not-augment-noise', action='store_true', help="use noise for data augmetnation, default True")
parser.add_argument('--not-augment-light', action='store_true', help="use bright & contrast for data augmetnation, default True")
parser.add_argument('--bright', default=10, type=int, help="bright value for bright & contrast augmentation, defaul 10")
parser.add_argument('--contrast', default=10, type=int, help="contrast value for bright & contrast augmentation, defaul 10")
parser.add_argument('--gt-mode', type=str, default='same', help="mode for generation of ground thruth  ['same', 'face', 'knn'] (default 'same')")
parser.add_argument('--model', type=str, default='mcnn1', help="network model  ['mcnn1', 'mcnn2', 'mcnn3', 'mcnn4'] (default 'mcnn-1')")

# Optimization options
parser.add_argument('--max-epoch', default=1000, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    help="initial learning rate")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size (default 32)")
# Miscs
parser.add_argument('--seed', type=int, default=64678, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")
parser.add_argument('--save-dir', type=str, default='log', help="path where results for each part/fold are saved")
parser.add_argument('--units', type=str, default='', help="folds/parts units to be trained, be default all folds/parts are trained")
parser.add_argument('--augment-only', action='store_true', help="run only data augmentation, default False")
parser.add_argument('--evaluate-only', action='store_true', help="run only data validation, --resume arg is needed, default False")
parser.add_argument('--train-only', action='store_true', help="train only, default False")
parser.add_argument('--skip-dir-check', action='store_true', help="skip directory validation and assumes that all data is already prepared, default False")
parser.add_argument('--save-plots', action='store_true', help="save plots of density map estimation (done only in test step), default False")
parser.add_argument('--den-scale-factor', type=float, default=1e3, help="scale factor to increasse small values in density maps")

args = parser.parse_args()

def train(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    output_dir_model = osp.join(output_dir, 'models')
    mkdir_if_missing(output_dir_model)
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

    # load net
    net = CrowdCounter(model = args.model)
    if not args.resume :
        network.weights_normal_init(net, dev=0.01)

    else:
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
            f = open(osp.join(resume_dir, "best_mae.bin"), "rb")
            best_mae = pickle.load(f)
            print("Best MAE: {0:.4f}".format(best_mae))
            f.close()
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

        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val, model = args.model, save_test_results=args.save_plots, plot_save_dir=osp.join(output_dir, 'plot-results-train/'), den_scale_factor = args.den_scale_factor)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(train_test_unit.to_string(),dataset_name,epoch)
            network.save_net(os.path.join(output_dir, "best_model.h5"), net)
            f = open(os.path.join(output_dir, "best_mae.bin"), "wb")
            pickle.dump(best_mae, f)
            f.close()


        print("Epoch: {0}, MAE: {1:.4f}, MSE: {2:.4f}, loss: {3:.4f}".format(epoch, mae, mse, train_loss))
        print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}".format(best_mae, best_mse, best_model))

def test(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    sys.stdout = Logger(osp.join(output_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    val_path =train_test_unit.test_dir_img
    val_gt_path = train_test_unit.test_dir_den

    if not args.resume :
        pretrained_model = osp.join(output_dir, 'best_model.h5')
    else:
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
    print("Using {} for testing.".format(pretrained_model))

    data_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size=1)
    mae,mse = evaluate_model(pretrained_model, data_loader, model = args.model, save_test_results=args.save_plots, plot_save_dir=osp.join(output_dir, 'plot-results-test/'), den_scale_factor = args.den_scale_factor)

    print("MAE: {0:.4f}, MSE: {1:.4f}".format(mae, mse))

def main():
    #augment data

    force_create_den_maps = True if args.force_den_maps else False
    force_augmentation = True if args.force_augment else False
    augment_noise = False if args.not_augment_noise else True 
    augment_light = False if args.not_augment_light else True
    augment_only = True if args.augment_only else False
    train_only = True if args.train_only else False
    skip_dir_check = True if args.skip_dir_check else False

    dataset = dataset_loader.init_dataset(name=args.dataset
    , force_create_den_maps = force_create_den_maps
    , force_augmentation = force_augmentation
    #sliding windows params
    , gt_mode = args.gt_mode
    , displace = args.displace
    , size_x= args.size_x
    , size_y= args.size_y
    , people_thr = args.people_thr
    #noise_params 
    , augment_noise = augment_noise
    #light_params
    , augment_light = augment_light
    , bright = args.bright
    , contrast = args.contrast
    #skip some directory validation
    , skip_dir_check = args.skip_dir_check
    )

    if augment_only:
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        print("Augment only - network will not be trained")
        return

    metadata = "_".join([args.dataset, dataset.signature()])
    out_dir_root = osp.join(args.save_dir, metadata)

    if args.units != '':
        units_to_train = [name.strip() for name in args.units.split(',')]
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        set_units = set(set_units)
        for unit in units_to_train:
            if not unit in set_units:
                raise RuntimeError("Invalid '{}' train-test unit".format(unit))
    else:
        units_to_train = [unit.metadata['name'] for unit in dataset.train_test_set]
    units_to_train = set(units_to_train)
    for train_test in dataset.train_test_set:
        if train_test.metadata['name'] in units_to_train:
            if args.evaluate_only:
                print("Testing {}".format(train_test.metadata['name']))
                test(train_test, out_dir_root)
            else:
                print("Training {}".format(train_test.metadata['name']))
                train(train_test, out_dir_root)
                if not(train_only):
                    print("Testing {}".format(train_test.metadata['name']))
                    test(train_test, out_dir_root)

if __name__ == '__main__':
    main()    

