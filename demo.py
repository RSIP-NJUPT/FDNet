# -*- coding:utf-8 -*-
# @Time       :2023/12/18 20:36 AM
# @AUTHOR     :Duo Wang
# @FileName   :demo.py
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import os

import argparse
import logging
from models.FDNet import MFFT
from utils import (trPixel2Patch, tsPixel2Patch, set_seed,
                   output_metric, train_epoch, valid_epoch, draw_classification_map)

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser(description="Training for FDNet")
parser.add_argument('--gpu_id', default='0',
                    help='gpu id')
parser.add_argument('--seed', type=int, default=0,
                    help='number of seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epoch')
parser.add_argument('--dataset', choices=['Houston', 'Augsburg', 'Muufl', 'Dafeng'], default='Dafeng',
                    help='dataset to use')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--flag', choices=['train', 'test'], default='test',
                    help='testing mark')
parser.add_argument('--patch_size', type=int, default=8,
                    help='cnn input size')
parser.add_argument('--wavename', type=str, choices=['bior2.2', 'db2', 'db4', 'db6'], default='db2',
                    help='type of wave')
parser.add_argument('--attn_kernel_size', type=int, default=9,
                    help='')
parser.add_argument('--coefficient_hsi', type=float, default=0.7,
                    help='weight of HSI data in feature fusion')
parser.add_argument('--vit_embed_dim', type=int, default=64,
                    help='number of channels in vit input data')
parser.add_argument('--vit_depth', type=int, default=1,
                    help='depth of vit')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# -------------------------------------------------------------------------------
# create log
logger = logging.getLogger("Trainlog")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler("cls_logs/{}/{}_{}.log".format(args.dataset, args.flag, args.dataset))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------------------------------------------------------------
def train1time():
    # -------------------------------------------------------------------------------
    if args.dataset == 'Houston':
        num_classes = 15
        DataPath1 = 'Data/Houston/Houston.mat'
        DataPath2 = 'Data/Houston/LiDAR.mat'
        LabelPath = 'Data/Houston/train_test_gt.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
    elif args.dataset == 'Augsburg':
        num_classes = 7
        DataPath1 = 'Data/Augsburg/data_HS_LR.mat'
        DataPath2 = 'Data/Augsburg/data_DSM.mat'
        Data1 = loadmat(DataPath1)['data_HS_LR']
        Data2 = loadmat(DataPath2)['data_DSM']
        LabelPath = 'Data/Augsburg/train_test_gt.mat'
    elif args.dataset == 'Muufl':
        num_classes = 11
        DataPath1 = 'Data/Muufl/HSI.mat'
        DataPath2 = 'Data/Muufl/LiDAR_DEM1.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        LabelPath = 'Data/Muufl/train_test_gt_60.mat'
    elif args.dataset == 'Dafeng':
        num_classes = 9
        DataPath1 = 'Data/Dafeng/HSI_30ch.mat'
        DataPath2 = 'Data/Dafeng/lidar_DSM_45.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['lidar']
        LabelPath = 'Data/Dafeng/train_test_gt_20.mat'
    else:
        raise "Requires correct dataset name!"

    Data1 = Data1.astype(np.float32)  # hsi
    Data2 = Data2.astype(np.float32)  # lidar
    TrLabel = loadmat(LabelPath)['train_data']
    TsLabel = loadmat(LabelPath)['test_data']

    patchsize = args.patch_size  # input spatial size for CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    TrainPatch1, TrainPatch2, TrainLabel = trPixel2Patch(
        Data1, Data2, patchsize, pad_width, TrLabel)
    TestPatch1, TestPatch2, TestLabel, _, _ = tsPixel2Patch(
        Data1, Data2, patchsize, pad_width, TsLabel)

    train_dataset = Data.TensorDataset(
        TrainPatch1, TrainPatch2, TrainLabel)
    test_dataset = Data.TensorDataset(
        TestPatch1, TestPatch2, TestLabel)
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    [H1, W1, _] = np.shape(Data1)
    Data2 = Data2.reshape([H1, W1, -1])
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    logger.info('\n')
    logger.info("=" * 50)
    logger.info("=" * 50)
    logger.info("hsi_height={0},hsi_width={1},hsi_band={2}".format(height1, width1, band1))
    logger.info("lidar_height={0},lidar_width={1},lidar_band={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = MFFT(l1=band1, l2=band2, patch_size=args.patch_size, num_classes=num_classes,
                 wavename=args.wavename, attn_kernel_size=args.attn_kernel_size, coefficient_hsi=args.coefficient_hsi,
                 vit_embed_dim=args.vit_embed_dim, deform_vit_depth=args.vit_depth)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # -------------------------------------------------------------------------------
    # train & test
    if args.flag == 'train':
        get_ts_result = False
        logger.info("start training")
        tic = time.time()
        for epoch in range(args.epochs):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(
                model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            logger.info("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                        .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

        torch.save(model.state_dict(), 'cls_param/{}/FDNet_{}.pkl'.format(args.dataset, args.dataset))
        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('cls_param/{}/FDNet_{}.pkl'.format(args.dataset, args.dataset)))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        logger.info("Final records:")
        logger.info("Maximal Accuracy: %f" % OA)
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Running Time: {:.2f}".format(toc - tic))

    if args.flag == 'test':
        # test best model
        get_ts_result = False
        tic_ts = time.time()
        model.eval()
        model.load_state_dict(torch.load('cls_param/{}/FDNet_{}.pkl'.format(args.dataset, args.dataset)))

        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        toc_ts = time.time()
        logger.info("Test records:")
        logger.info("Maximal Accuracy: %f" % OA)
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Testing Time: {:.2f}".format(toc_ts - tic_ts))
        logger.info("Parameter:")
        logger.info(vars(args))

        # draw map
        if args.dataset == 'Houston':
            TR_TS_Path = 'Data/Houston/tr_ts.mat'
        elif args.dataset == 'Augsburg':
            TR_TS_Path = 'Data/Augsburg/tr_ts.mat'
        elif args.dataset == 'Muufl':
            TR_TS_Path = 'Data/Muufl/tr_ts.mat'
        elif args.dataset == 'Dafeng':
            TR_TS_Path = 'Data/Dafeng/tr_ts.mat'
        else:
            raise "Correct dataset needed!"

        TR_TS_Label = loadmat(TR_TS_Path)['tr_ts']
        # draw gt map
        draw_classification_map(TR_TS_Label, 'cls_map/{}/{}_groundTruth.png'.format(args.dataset, args.dataset),
                                args.dataset)
        # draw cls map
        TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label, xIndex_list, yIndex_list = tsPixel2Patch(
            Data1, Data2, patchsize, pad_width, TR_TS_Label)
        TR_TS_dataset = Data.TensorDataset(
            TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label)
        best_test_loader = Data.DataLoader(
            TR_TS_dataset, batch_size=args.batch_size, shuffle=False)

        get_ts_result = True  # if True, return cls result
        ts_result = valid_epoch(model, best_test_loader, criterion, get_ts_result)
        ts_result_matrix = np.full((H1, W1), 0)
        for i in range(len(ts_result)):
            ts_result_matrix[xIndex_list[i], yIndex_list[i]] = ts_result[i]
        draw_classification_map(ts_result_matrix, 'cls_map/{}/{}_predLabel.png'.format(args.dataset, args.dataset),
                                args.dataset)


if __name__ == '__main__':
    set_seed(args.seed)
    train1time()
