#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import sys
import random
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

import cv2 
import shutil


# In[14]:


def Write_TXT(txt_file, data_list):
    fd  = open(txt_file, 'w')
    for data in data_list:
        fd.write(data)
        fd.write('\n')
    fd.close()

def parse_xml(xml_path):
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
        
        xc = (xmin+xmax)/2/width 
        yc = float(ymin+ymax)/2/height 
        w  = (xmax-xmin)/width
        h = (ymax-ymin)/height
        
        bboxs.append((xc,yc,w,h))

    return bboxs


# In[15]:


fn = './detector/gun/Train/Annotations/ia_100000004463.xml'
bboxs = parse_xml(fn)
print(bboxs)


# In[89]:


TRAIN_PATH = './detector/gun/Train/JPEGImages/'
base_images = glob.glob(TRAIN_PATH + '*.jpg')
random.shuffle(base_images)
print(len(base_images))

train_list = []

for idx, img in enumerate(tqdm(base_images)):

    imgname = img.split('/')[-1]
    imgbasename = imgname.split('.')[0]
    
    print("processing {}".format(imgname ))
    
    anno_fn = '../Annotations/{}.xml'.format(imgbasename)
    annofile = '{}.txt'.format(imgbasename)
    imgpath =  os.path.abspath(img)
    
    #print(anno_fn)
    
    labels = str(); 
    bboxs = parse_xml(anno_fn)
    for bbox in bboxs:
        labels = labels + '0 {} {} {} {}\n'.format(bbox[0],bbox[1],bbox[2],bbox[3])
    #print(labels)
    
    #shutil.copyfile(imgpath,'/media/ubuntu/MyHDataStor1/work/products/LPR/ultralytics_yolo3/data/Gun/images/{}'.format(imgname))
    
    des_fn = '/media/ubuntu/MyHDataStor1/work/data/Gun/images/{}'.format(imgname)
    sl_cmd = 'ln -s {} {}'.format(imgpath, des_fn)
    print(sl_cmd)
    os.system(sl_cmd)
    
    annofile = '/media/ubuntu/MyHDataStor1/work/data/Gun/labels/'+annofile
    fp = open(annofile, 'wt')
    fp.write(labels)
    fp.close() 

    train_list.append('/media/ubuntu/MyHDataStor1/work/data/Gun/images/{}'.format(imgname) )

Write_TXT('/media/ubuntu/MyHDataStor1/work/data/Gun_train.txt', train_list)


# In[75]:


TEST_PATH = './detector/gun/Test/JPEGImages/'
base_images = glob.glob(TEST_PATH + '*.jpg')
random.shuffle(base_images)
print(len(base_images))
test_list = []

for idx, img in enumerate(tqdm(base_images)):

    imgname = img.split('/')[-1]
    imgbasename = imgname.split('.')[0]
    
    print("processing {}".format(imgname ))
    
    anno_fn = '../Annotations/{}.xml'.format(imgbasename)
    annofile = '{}.txt'.format(imgbasename)
    imgpath =  os.path.abspath(img)
    
    #print(anno_fn)
    
    labels = str(); 
    bboxs = parse_xml(anno_fn)
    for bbox in bboxs:
        labels = labels + '0 {} {} {} {}\n'.format(bbox[0],bbox[1],bbox[2],bbox[3])
    #print(labels)
    
    #shutil.copyfile(imgpath,'/media/ubuntu/MyHDataStor1/work/products/LPR/ultralytics_yolo3/data/Gun/images/{}'.format(imgname))
    
    des_fn = '/media/ubuntu/MyHDataStor1/work/data/Gun/images/{}'.format(imgname)
    sl_cmd = 'ln -s {} {}'.format(imgpath, des_fn)
    print(sl_cmd)
    os.system(sl_cmd)
    
    annofile = '/media/ubuntu/MyHDataStor1/work/data/Gun/labels/'+annofile
    fp = open(annofile, 'wt')
    fp.write(labels)
    fp.close() 

    test_list.append('/media/ubuntu/MyHDataStor1/work/data/Gun/images/{}'.format(imgname) )

Write_TXT('/media/ubuntu/MyHDataStor1/work/data/Gun_test.txt', train_list)


# In[ ]:




