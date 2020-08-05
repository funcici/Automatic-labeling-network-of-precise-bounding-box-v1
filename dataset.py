import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random
import torchvision.transforms.functional as TF




#re_size = (256, 256)
#cr_size = (224, 224)



class ImageDataTrain(data.Dataset):
    def __init__(self):
        self.img_folder = './image'
        self.label_folder = './label'
        self.edge_folder = './edge'

        self.img_list = os.listdir(self.img_folder)
        ###self.sal_num = len(self.sal_list)

    def __getitem__(self, item):

        flip_flag = random.randint(0, 360)
        sal_image = load_image(os.path.join(self.img_folder, self.img_list[item] ),flip_flag)

        sal_label = load_sal_label(os.path.join(self.label_folder, self.img_list[item] ),flip_flag)

        sal_edge = load_edge_label(os.path.join(self.edge_folder, self.img_list[item] ),flip_flag)

        #sal_image, sal_label, sal_edge = cv_random_flip(sal_image, sal_label, sal_edge)

        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)
        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample

    def __len__(self):
        # return max(max(self.edge_num, self.sal_num), self.skel_num)
        return len(self.img_list)



class ImageDataTest(data.Dataset):
    def __init__(self, test_mode=1, sal_mode='e'):
        if test_mode == 0:


            self.test_fold = './test'
            self.img_folder = './cut_image1'
            self.image_list = os.listdir(self.img_folder)

        '''
        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]
        '''
        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.img_folder, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item%self.image_num], 'size': im_size}

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        # return max(max(self.edge_num, self.skel_num), self.sal_num)
        return len(self.image_list)


# get the dataloader (Note: without data augmentation, except saliency with random flip)
def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
    else:
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset

def load_image(pah,flip_flag):
    if not os.path.exists(pah):
        print('File Not Exists')

    im = Image.open(pah)
    im = TF.rotate(img=im, angle=flip_flag, resample=Image.NEAREST)
    #im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    #in_ = cv2.resize(in_, im_sz, interpolation=cv2.INTER_CUBIC)
    # in_ = in_[:,:,::-1] # only if use PIL to load image
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    #im = cv2.imread(pah)
    im = Image.open(pah)

    print(pah)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    #in_ = cv2.resize(in_, (224, 224), interpolation=cv2.INTER_LINEAR)
    # in_ = in_[:,:,::-1] # only if use PIL to load image
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_edge_label(pah,flip_flag):
    """
    pixels > 0.5 -> 1
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    im = TF.rotate(img=im, angle=flip_flag, resample=Image.NEAREST)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label

def load_skel_label(pah,flip_flag):
    """
    pixels > 0 -> 1
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    im = TF.rotate(img=im, angle=flip_flag, resample=Image.NEAREST)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label[np.where(label > 0.)] = 1.
    label = label[np.newaxis, ...]
    return label

def load_sal_label(pah,flip_flag):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    im = TF.rotate(img=im, angle=flip_flag, resample=Image.NEAREST)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def load_sem_label(pah):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = Image.open(pah)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    # label = label / 255.
    label = label[np.newaxis, ...]
    return label

def edge_thres_transform(x, thres):
    # y0 = torch.zeros(x.size())
    y1 = torch.ones(x.size())
    x = torch.where(x >= thres, y1, x)
    return x

def skel_thres_transform(x, thres):
    y0 = torch.zeros(x.size())
    y1 = torch.ones(x.size())
    x = torch.where(x > thres, y1, y0)
    return x



def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 360)

    img = TF.rotate(img=img, angle=flip_flag, resample=Image.NEAREST)
    label = TF.rotate(img=label, angle=flip_flag, resample=Image.NEAREST)
    edge = TF.rotate(img=edge, angle=flip_flag, resample=Image.NEAREST)
    #edge = edge[:,:,::-1].copy()
    return img, label, edge





def cv_random_crop_flip(img, label, resize_size, crop_size, random_flip=True):
    def get_params(img_size, output_size):
        h, w = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    if random_flip:
        flip_flag = random.randint(0, 1)
    img = img.transpose((1,2,0)) # H, W, C
    label = label[0,:,:] # H, W
    img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
    i, j, h, w = get_params(resize_size, crop_size)
    img = img[i:i+h, j:j+w, :].transpose((2,0,1)) # C, H, W
    label = label[i:i+h, j:j+w][np.newaxis, ...] # 1, H, W
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label

def random_crop(img, label, size, padding=None, pad_if_needed=True, fill_img=(123, 116, 103), fill_label=0, padding_mode='constant'):

    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    if padding is not None:
        img = F.pad(img, padding, fill_img, padding_mode)
        label = F.pad(label, padding, fill_label, padding_mode)

    # pad the width if needed
    if pad_if_needed and img.size[0] < size[1]:
        img = F.pad(img, (int((1 + size[1] - img.size[0]) / 2), 0), fill_img, padding_mode)
        label = F.pad(label, (int((1 + size[1] - label.size[0]) / 2), 0), fill_label, padding_mode)
    # pad the height if needed
    if pad_if_needed and img.size[1] < size[0]:
        img = F.pad(img, (0, int((1 + size[0] - img.size[1]) / 2)), fill_img, padding_mode)
        label = F.pad(label, (0, int((1 + size[0] - label.size[1]) / 2)), fill_label, padding_mode)

    i, j, h, w = get_params(img, size)
    return [F.crop(img, i, j, h, w), F.crop(label, i, j, h, w)]
