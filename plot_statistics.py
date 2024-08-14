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
        bboxs.append((xmin,ymin,xmax-xmin,ymax-ymin,width,height))

    return bboxs



TRAIN_PATH = './detector/gun/Train/JPEGImages/'
base_images = glob.glob(TRAIN_PATH + '*.jpg')
random.shuffle(base_images)
print(len(base_images))


from matplotlib import pyplot as plt
from matplotlib import patches as patch


area_image = []
area_gun = [] 
num_gun = [] 


import pickle
import os 

fn_stat = 'gun-statistic.pkl'

if not os.path.exists(fn_stat):
  for idx, img in enumerate(tqdm(base_images)):
    
    #print("idx = {} processing image {}".format(idx,img))
   
    tmp = img.split('/')
    imgname = tmp[-1]
    imgbasename = imgname.split('.')[0]
    
    annodir =  '/'.join(tmp[:-2])+'/Annotations'
    annofn = annodir +  '/{}.xml'.format(imgbasename)
    #print("annofn = {}".format(annofn)) 
    #absimgpath =  os.path.abspath(img)

    try: 
      bboxs = parse_xml(annofn)
      # Create a Rectangle patch
      for b in bboxs:
        x1 = b[0]
        y1 = b[1] 
        w = b[2]
        h = b[3] 
        w_img = b[4] 
        h_img = b[5] 
        area_gun.append(w*h) 
        
      area_image.append(w_img*h_img)  
      num_gun.append(len(bboxs))  
    except: 
      print("annotation file {} fails".format(annofn))

  
  with open(fn_stat, 'wb') as fp:
    pickle.dump([area_image,area_gun,num_gun], fp)

else: 

  with open(fn_stat,'rb') as fp:
    [area_image,area_gun,num_gun] = pickle.load(fp)


#print(area_image)
#print(area_gun) 
#print(num_gun) 

def plot_hist_bar(ax,hists,bins): 

  plt = ax 
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  plt.bar(center, hists, align='center', width=width)
  plt.grid(axis='y', alpha=0.75)

import numpy as np 

fig = plt.figure()

area_image = np.asarray(area_image) 
area_gun = np.asarray(area_gun) 
num_gun = np.asarray(num_gun) 


area_image_max = np.max(area_image)
area_image_min = np.min(area_image) 
hists,bins = np.histogram(area_image, bins=50, range=[area_image_min,area_image_min*100])
#hist = hist/np.sum(hist) 
ax = fig.add_subplot(3,1,1)
plot_hist_bar(ax,hists,bins) 
ax.set_ylabel('Number of Images')
ax.set_xlabel('Image width x height')

area_image_max = np.max(area_gun)
area_image_min = np.min(area_gun) 
hists,bins = np.histogram(area_gun, bins=50, range=[area_image_min,area_image_min*100])
#hist = hist/np.sum(hist) 
ax = fig.add_subplot(3,1,2)
plot_hist_bar(ax,hists,bins) 
ax.set_ylabel('Number of Images')
ax.set_xlabel('Gun width x height')

area_image_max = np.max(num_gun)
area_image_min = np.min(num_gun) 
hists,bins = np.histogram(num_gun, bins=10, range=[1,10])
#hist = hist/np.sum(hist) 
ax = fig.add_subplot(3,1,3)
plot_hist_bar(ax,hists,bins) 
ax.set_ylabel('Number of Images')
ax.set_xlabel('Number of Guns per Image')


plt.show() 
 




