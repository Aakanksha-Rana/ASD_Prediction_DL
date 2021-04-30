# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

import lrs

from data_loader import AVADataset

from model import *
from torchvision.utils import make_grid, save_image
##### Additions for gradcam
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()])
     
    test_transform = transforms.Compose([
        transforms.ToTensor()])
    trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=train_transform)
    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
        shuffle=False, num_workers=config.num_workers)

#    base_model = models.vgg16(pretrained=True)
#    base_model = models.resnet18(pretrained=True)
    base_model = models.resnet101(pretrained=True, progress = False)
#    base_model = models.inception_v3(pretrained=True)
    model = NIMA(base_model)
#    model = NIMA()
    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )
#    optimizer = optim.Adam( model.parameters(), lr = conv_base_lr, betas=(0.9,0.999))
# Loss functions  
#    criterion = torch.nn.L1Loss()
    criterion = torch.nn.CrossEntropyLoss()
    
    # send hyperparams
    lrs.send({
        'title': 'EMD Loss',
        'train_batch_size': config.train_batch_size,
        'val_batch_size': config.val_batch_size,
        'optimizer': 'SGD',
        'conv_base_lr': config.conv_base_lr,
        'dense_lr': config.dense_lr,
        'momentum': 0.9
        })

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))


    if config.test:
#        start.record()
        print('Testing')  
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        target_layer = model.features
        # compute mean score
        test_transform = test_transform#val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.test_img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

        ypreds =  []
        ylabels =  []
        im_ids = []
#        std_preds = []
        count = 0
        gradcam = GradCAM(model, target_layer)
        
        for data in test_loader:
            im_id = data['img_id']
            im_name = os.path.split(im_id[0])
            myname= os.path.splitext(im_name[1])
            image = data['image'].to(device)
            mask, _ = gradcam(image)
            heatmap, result = visualize_cam(mask, image)
            im = transforms.ToPILImage()(result)
            im.save(myname[0]+".jpg")
            labels = data['annotations'].to(device).long()
            output = model(image)
            output = output.view(-1, 2)
            bpred = output.to(torch.device("cpu"))
            cpred = bpred.data.numpy()
            blabel = labels.to(torch.device("cpu"))
            clabel = blabel.data.numpy()
#            predicted_mean, predicted_std = 0.0, 0.0
#            for i, elem in enumerate(output, 1):
#                predicted_mean += i * elem
#            for j, elem in enumerate(output, 1):
#                predicted_std += elem * (i - predicted_mean) ** 2
            ypreds.append(cpred)
            ylabels.append(clabel)
            im_name = os.path.split(im_id[0])
            im_ids.append(im_name[1])
            count= count+1
        np.savez('Test_results_16.npz' , Label = ylabels, Predict = ypreds)
        df = pd.DataFrame(data={'Label': ylabels, "Predict": ypreds})
        print(df.dtypes)
        df.to_pickle("./Test_results_19_resnet.pkl")  
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/path/to/train')
    parser.add_argument('--val_img_path', type=str, default='/path/to/val')
    parser.add_argument('--test_img_path', type=str, default='/path/to/test')
    parser.add_argument('--train_csv_file', type=str, default='./Train_final.csv')
    parser.add_argument('--val_csv_file', type=str, default='./Val_final.csv')
    parser.add_argument('--test_csv_file', type=str, default='./Test_final.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default = False)
    parser.add_argument('--test', type=bool, default = True)
    parser.add_argument('--conv_base_lr', type=float, default=.001)
    parser.add_argument('--dense_lr', type=float, default=.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='./ckpts/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=16)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    config = parser.parse_args()

    main(config)
