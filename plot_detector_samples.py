#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

import cv2 
import shutil


def Write_TXT(txt_file, data_list):
    fd  = open(txt_file, 'w')
    for data in data_list:
        fd.write(data)
        fd.write('\n')
    fd.close()

def parse_xml(fn):
    root = ET.parse(fn).getroot()
    filename = root.find('filename').text 
    width  = float(root.find('size').find('width').text)
    height = float(root.find('size').find('height').text)
    
    bboxs = []
    for obj in root.findall('object'):
        xmin = float(obj.find('bndbox').find('xmin').text)
        ymin = float(obj.find('bndbox').find('ymin').text)
        xmax = float(obj.find('bndbox').find('xmax').text)
        ymax = float(obj.find('bndbox').find('ymax').text)
        '''
        xc = (xmin+xmax)/2/width 
        yc = float(ymin+ymax)/2/height 
        w  = (xmax-xmin)/width
        h = (ymax-ymin)/height
        '''        
        bboxs.append((xmin,ymin,xmax-xmin,ymax-ymin))

    return bboxs


fn = './detector/gun/Train/Annotations/ia_100000004463.xml'
bboxs = parse_xml(fn)
print(bboxs)


TRAIN_PATH = './detector/gun/Train/JPEGImages/'
base_images = glob.glob(TRAIN_PATH + '*.jpg')
random.shuffle(base_images)
print(len(base_images))


from matplotlib import pyplot as plt
from matplotlib import patches as patch


fig = plt.figure()

for idx, img in enumerate(tqdm(base_images)):
    
    print("idx = {} processing image {}".format(idx,img))
    
    tmp = img.split('/')
    imgname = tmp[-1]
    imgbasename = imgname.split('.')[0]
    
    annodir =  '/'.join(tmp[:-2])+'/Annotations'
    annofn = annodir +  '/{}.xml'.format(imgbasename)

    print("annofn = {}".format(annofn)) 

    absimgpath =  os.path.abspath(img)
    
    bboxs = parse_xml(annofn)
    
    print("bboxs = {}".format(bboxs)) 
    
    img=cv2.imread(img)


    ax = fig.add_subplot(4,10,idx+1)
    ax.imshow(img) 

    # Create a Rectangle patch
    for b in bboxs:
        x1 = b[0]
        y1 = b[1] 
        w = b[2]
        h = b[3] 

        rect = patch.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
  
    
    if idx+1>=40:
        break

plt.axis('off')        
plt.show()


