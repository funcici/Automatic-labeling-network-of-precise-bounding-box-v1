# coding=utf-8
import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.backends import cudnn
from model import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import math
import time
import sys
import PIL.Image
import scipy.io
import os
import logging
import os
from PIL import Image
import pandas as pd
import csv
import utilize

EPSILON = 1e-8
p = OrderedDict()

from dataset import get_loader
base_model_cfg = 'resnet'
p['lr_bone'] = 5e-5  # Learning rate resnet:5e-5, vgg:2e-5
p['lr_branch'] = 0.025  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [100, 200] # [6, 9], now x3 #15
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 8
tmp_path = 'tmp_see'


class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.save_fold = save_fold
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.
        # inference: choose the side map (see paper)
        if config.visdom:
            self.visual = Viz_visdom("trueUnify", 1)
        self.build_model()
        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def get_params(self, base_lr):
        ml = []
        for name, module in self.net_bone.named_children():
            print(name)
            if name == 'loss_weight':
                ml.append({'params': module.parameters(), 'lr': p['lr_branch']})          
            else:
                ml.append({'params': module.parameters()})
        return ml

    # build the network
    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()
            
        self.net_bone.eval()  # use_global_stats = True
        self.net_bone.apply(weights_init)
        if self.config.mode == 'train':
            if self.config.load_bone == '':
                if base_model_cfg == 'vgg':
                    self.net_bone.base.load_pretrained_model(torch.load(self.config.vgg))
                elif base_model_cfg == 'resnet':
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
            if self.config.load_bone != '': self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])

        self.print_network(self.net_bone, 'trueUnify bone part')

    # update the learning rate
    def update_lr(self, rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate


    def test(self, test_mode=0):
        EPSILON = 1e-8
        img_num = len(self.test_loader)
        time_t = 0.0
        name_t = 'test/'

        if not os.path.exists(os.path.join(self.save_fold, name_t)):             
            os.mkdir(os.path.join(self.save_fold, name_t))
        for i, data_batch in enumerate(self.test_loader):
            self.config.test_fold = self.save_fold
            print(self.config.test_fold)
            images_, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            
            with torch.no_grad():
                
                images = Variable(images_)
                if self.config.cuda:
                    images = images.cuda()
                print(images.size())
                time_start = time.time()
                up_edge, up_sal, up_sal_f = self.net_bone(images)
                torch.cuda.synchronize()

                time_end = time.time()
                print(time_end - time_start)
                time_t = time_t + time_end - time_start
                print('0000000000000000000000000000000000000000000000000000000')
                print (torch.max(up_sal_f[-1]))
                print(torch.min(up_sal_f[-1]))
                print('0000000000000000000000000000000000000000000000000000000')
                pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())
                print(np.max(pred))
                print(np.min(pred))

                multi_fuse = (255/np.max(pred)) * pred
                '''
                c = np.sum(multi_fuse)/(224*224)

                for i in (range(multi_fuse.shape[0])):
                    for ii in (range(multi_fuse.shape[1])):
                        if multi_fuse[i][ii] > c:
                            multi_fuse[i][ii] = 255
                        else:
                            multi_fuse[i][ii] = 0
                 '''
                

                
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '.png'), multi_fuse)


        print("--- %s seconds ---" % (time_t))
        print('Test Done!')

    def application(self):
        h = 32
        w = 32
        longitudeoffset = self.config.longitudeoffset
        latitudeoffset = self.config.latitudeoffset
        pixel = self.config.pixel
        out_file = open('./file/new.vif', 'w', encoding='utf-8')

        # 读取文件头
        start = utilize.load_from_txt("./file/start.txt", encoding="utf-8")
        for line in start:
            line = line.replace("139.705710411072", str(longitudeoffset)).replace("35.5766701698303",str(latitudeoffset))
            out_file.write(line)  # first is old ,second is new
        # 读取block
        block = utilize.load_from_txt("./file/block.txt", encoding="utf-8")




        point = pd.read_csv(self.config.csv_dir)
        image_big0 = cv2.imread(self.config.image_big_dir)
        image_big = cv2.copyMakeBorder(image_big0, h, h, w, w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ii = 0
        boxs = []
        for i in point["name"]:
            print(i)
            xs = int(point['center_point_x'][ii])
            ys = int(point['center_point_y'][ii])
            xe = int(point['center_point_x'][ii]) + (2 * w)
            ye = int(point['center_point_y'][ii]) + (2 * h)
            print(xs, ys, xe, ye)
            xs_t = int(point['center_point_x'][ii]) -w
            ys_t = int(point['center_point_y'][ii]) -h
            ii +=1
            st = [xs_t,ys_t]
            im = image_big [ys:ye, xs:xe]
            im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_LINEAR)
            in_ = np.array(im, dtype=np.float32)
            # in_ = cv2.resize(in_, (224, 224), interpolation=cv2.INTER_LINEAR)
            in_ = in_[:,:,::-1].copy()
            in_ -= np.array((104.00699, 116.66877, 122.67892))
            in_ = in_.transpose((2, 0, 1))
            in_ = torch.Tensor(in_)
            in_ = in_.unsqueeze(0)
            with torch.no_grad():
                images = Variable(in_)

                if self.config.cuda:
                    images = images.cuda()
                print(images.size())
                time_start = time.time()
                up_edge, up_sal, up_sal_f = self.net_bone(images)
                torch.cuda.synchronize()
                pred = np.squeeze(torch.sigmoid(up_sal_f[-1]).cpu().data.numpy())
                multi_fuse = (255 / np.max(pred)) * pred
                multi_fuse[np.where(multi_fuse >=50)] = 255
                multi_fuse[np.where(multi_fuse < 50)] = 0
                p =[]
                for iii in (range(multi_fuse.shape[0])):
                    for iiii in (range(multi_fuse.shape[1])):
                        if multi_fuse[iii][iiii] ==255:
                            p.append([iiii,iii])
                p = np.array(p)
                #p = cv2.findContours(multi_fuse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                #print(p)
                rect = cv2.minAreaRect(p)
                # 得到最小矩形的坐标
                box = cv2.boxPoints(rect)
                # 标准化坐标到整数
                box = np.int0(box)
                cv2.imshow('a', multi_fuse)
                cv2.drawContours(im, [box], 0, (0, 0, 255), 1)
                cv2.imshow('b', im)
                cv2.waitKey(1)
                box = np.int0(box/2)
                box = box + st
                pos = [box[0],box[1],box[2],box[3]]
                for line in block[:-7]:
                    line = line.replace("矩形612", i)
                    out_file.write(line)  # first is old ,second is new

                for xy in pos:
                    init = "        <GeoShapePoint x=xxx y=yyy />\n"
                    geo_x, geo_y = utilize.xy2geo(xy, pixel, [longitudeoffset, latitudeoffset])
                    init = init.replace("xxx", "\"" + str(geo_x) + "\"").replace("yyy", "\"" + str(geo_y) + "\"")
                    out_file.write(init)  # first is old ,second is new

                for line in block[-3:]:
                    out_file.write(line)  # first is old ,second is new
            boxs.append(box)

        end = utilize.load_from_txt("./file/end.txt", encoding="utf-8")
        for line in end:
            out_file.write(line)  # first is old ,second is new

        for box in boxs:
            cv2.drawContours(image_big0, [box], 0, (0, 0, 255), 1)
        cv2.imshow('a', image_big0)
        cv2.imwrite('./result.jpg',image_big0)
        cv2.waitKey(0)
    # training phase



    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        F_v = 0
        if not os.path.exists(tmp_path): 
            os.mkdir(tmp_path)
        for epoch in range(self.config.epoch):                          
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge']
                if sal_image.size()[2:] != sal_label.size()[2:]:
                    print("Skip this batch")
                    continue

                sal_image, sal_label, sal_edge = Variable(sal_image), Variable(sal_label), Variable(sal_edge)
                if self.config.cuda: 
                    sal_image, sal_label, sal_edge = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda()

                up_edge, up_sal, up_sal_f = self.net_bone(sal_image)
                # edge part
                edge_loss = []
                for ix in up_edge:
                    edge_loss.append(bce2d_new(ix, sal_edge, reduction='sum'))
                edge_loss = sum(edge_loss) / (nAveGrad * self.config.batch_size)
                r_edge_loss += edge_loss.data
                # sal part
                sal_loss1= []
                sal_loss2 = []
                for ix in up_sal:
                    sal_loss1.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))

                for ix in up_sal_f:
                    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, sal_label, reduction='sum'))
                sal_loss = (sum(sal_loss1) + sum(sal_loss2)) / (nAveGrad * self.config.batch_size)
              
                r_sal_loss += sal_loss.data
                loss = sal_loss + edge_loss
                r_sum_loss += loss.data
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
       
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()           
                    aveGrad = 0


                if i % showEvery == 0:

                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,  r_edge_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sal_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sum_loss*(nAveGrad * self.config.batch_size)/showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

                if i % 8 == 0:

                    vutils.save_image(torch.sigmoid(up_sal_f[-1].data), tmp_path+'/iter%d-sal-0.jpg' % i , normalize=True, padding = 0)#% i
                    vutils.save_image(sal_image.data.cpu().float(), tmp_path+'/iter%d-sal-data.jpg' % i,normalize=True, padding = 0)
                    vutils.save_image(sal_label.data, tmp_path+'/iter%d-sal-target.jpg' % i, padding = 0)

            
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))
                
            if epoch in lr_decay_epoch:
                self.lr_bone = self.lr_bone * 0.1  
                self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])
        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
        
def bce2d_new(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

