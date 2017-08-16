#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

"""
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
"""

CLASSES = ('__background__','face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'myvgg': ('VGG_CNN_M_1024',
                  'VGG_CNN_M_1024_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'myzf': ('ZF',
                  'zf_rpn_stage1_iter_80000.caffemodel'),
}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        write_file.write(array[current_image] + ' ') 
        #add by zhipeng
        #write_file.write('face' + ' ')
        ##########   add by zhipeng for write rectange to txt   ########
        #bbox[0]:x, bbox[1]:y, bbox[2]:x+w, bbox[3]:y+h
        write_file.write( "{} {} {} {}\n"
                                .format(str(int(bbox[0])), str(int(bbox[1])),
                                str(int(bbox[2])-int(bbox[0])),
                                str(int(bbox[3])-int(bbox[1]))))
        #print "zhipeng, bbox:", bbox, "score:",score
        ##########   add by zhipeng for write rectange to txt   ########



def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    '''''im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']'''

    read_file_name = '/home/xiao/code/py-faster-rcnn-master/py-faster-rcnn/data/pos_fold/name.txt'
    write_file_name = '/home/xiao/code/py-faster-rcnn-master/py-faster-rcnn/data/pos_fold/annotate.txt'
    write_file = open(write_file_name, "w")

    with open(read_file_name, "r") as ins:
        array = []
        for line in ins:
            line = line[:-1]
            array.append(line)      # list of strings

    number_of_images = len(array)

    for current_image in range(number_of_images):
        if current_image % 100 == 0:
            print 'Processing image : ' + str(current_image)
        # load image and convert to gray
        read_img_name = '/home/xiao/code/py-faster-rcnn-master/py-faster-rcnn/data/pos/' + array[current_image].rstrip()
        #write_file.write(array[current_image]) #add by zhipeng
        demo(net, read_img_name)

    write_file.close()

    '''''for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        write_file.write(im_name + '\n') #add by zhipeng
        demo(net, im_name)'''

    #write_file.close()  # add by zhipeng,close file
    plt.show()  