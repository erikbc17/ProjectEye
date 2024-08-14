#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

import cv2 
import shutil


# In[2]:


TRAIN_PATH = './classifier/gun/train/'
base_images = glob.glob(TRAIN_PATH + '*.jpg')
random.shuffle(base_images)
print(len(base_images))


# In[3]:


from matplotlib import pyplot as plt


# In[9]:


for idx, img in enumerate(tqdm(base_images)):
   
    
    img=cv2.imread(img)

    plt.subplot(4,15,idx+1),plt.imshow(img)
   
    
    if idx+1>=60:
        break
        
plt.show()


# In[ ]:




