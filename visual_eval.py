import cv2
import torch
from vi3o.image import imview

from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.backends.cudnn as cudnn
import numpy as np

from ssd import build_ssd

cudnn.benchmark = True

num_classes = len(labelmap) + 1                      # +1 for background
net = build_ssd('test', 300, num_classes)            # initialize SSD
net.load_state_dict(torch.load("weights/ssd300_COCO_5000.pth"))
net.cuda()
net.eval()

dataset_mean = (104, 117, 123)
set_type = 'test'
dataset = VOCDetection(VOC_ROOT, [('2007', set_type)],
                       BaseTransform(300, dataset_mean),
                       VOCAnnotationTransform())
num_images = len(dataset)

for i in range(num_images):
    im, gt, h, w = dataset.pull_item(i)

    img = im.permute(1,2,0).numpy().copy()

    if False:
        for xmin, ymin, xmax, ymax, label_ind in gt:
            print(labelmap[int(label_ind)], img.shape)
            cv2.rectangle(img,
                          (int(xmin * 300), int(ymin * 300)),
                          (int(xmax * 300), int(ymax * 300)),
                          (255,0,0), 2)
    else:
        detections = net(im[None,:,:,:].cuda()).data
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                xmin, ymin, xmax, ymax = (detections[0, i, j, 1:]).cpu().numpy()
                cv2.rectangle(img,
                              (int(xmin * 300), int(ymin * 300)),
                              (int(xmax * 300), int(ymax * 300)),
                              (255,0,0), 2)
                print(labelmap[i - 1])
                j += 1


        # print(rect)
    imview(img)