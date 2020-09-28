import os
import os.path
import sys
import random
import torch
import torch.utils.data as data
import cv2
import numpy as np
from copy import deepcopy

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None, cutmix_prob=0.):
        self.preproc = preproc
        self.cutmix_prob = cutmix_prob
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        if random.random() < self.cutmix_prob:
            return self.cutmix(index)
        else:
            return self.load_img_target(index)

    def cutmix(self, index):
        img_dim = self.preproc.img_dim
        w, h = img_dim, img_dim
        s = img_dim // 2
        n = len(self.imgs_path)
        r_index = random.randint(0, n - 1)
        
        image, boxes = self.load_img_target(index)
        image = image.numpy().transpose(1,2,0)#.astype(np.int32)
        boxes = deepcopy(boxes * img_dim)
        
        r_image, r_boxes = self.load_img_target(r_index)
        r_image = r_image.numpy().transpose(1,2,0)#.astype(np.int32)
        r_boxes = deepcopy(r_boxes * img_dim)
    
        xc, yc = [int(random.uniform(img_dim * 0.4, img_dim * 0.6)) for _ in range(2)]  # center x, y
        direct = random.randint(0, 3)

        result_image = image.copy()
        result_boxes = []

        #for i, index in enumerate(indexes):
        if True:
            if direct == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif direct == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif direct == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif direct == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            #print(result_image.shape, image.shape)
            padw = x1a - x1b
            padh = y1a - y1b

            r_boxes[:, 0] += padw
            r_boxes[:, 1] += padh
            r_boxes[:, 2] += padw
            r_boxes[:, 3] += padh
            r_boxes[:, 4] += padw
            r_boxes[:, 5] += padh
            r_boxes[:, 6] += padw
            r_boxes[:, 7] += padh
            r_boxes[:, 8] += padw
            r_boxes[:, 9] += padh
            r_boxes[:, 10] += padw
            r_boxes[:, 11] += padh
            r_boxes[:, 12] += padw
            r_boxes[:, 13] += padh
            
            result_boxes.append(r_boxes)
            
            result_boxes = np.concatenate(result_boxes, 0)
            np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
            result_boxes = result_boxes.astype(np.int32)
            result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
            
            result_image[y1a:y2a, x1a:x2a] = (result_image[y1a:y2a, x1a:x2a] + r_image[y1b:y2b, x1b:x2b]) / 2
            
        result_boxes = (np.concatenate([result_boxes,boxes]) / img_dim) #.astype(np.int32)

        return torch.from_numpy(result_image.transpose(2,0,1)), result_boxes

    def load_img_target(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
