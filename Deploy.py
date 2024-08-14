
'''Importing the Libraries'''

import os
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array

model = load_model('./Model/')
# Here we are using OpenCV to operate webcam. 
# We are passing the frames to the trained model and calculating the prediction confidence. 

webcam = cv2.VideoCapture(0) # Selecting the primary webcam
classes = ['gun_found', 'no_gun_found']

while webcam.isOpened():

  # Image processing for gun detection:

  stat,frm = webcam.read() # Getting the frame from webcam
  x = cv2.resize(frm, (100,100)) # Resizing the frame
  x = x.astype('float')/255 # converting to greyscale
  x = img_to_array(x) # Converting Greyscale image to array
  x = np.expand_dims(x, axis=0)

  # Label for the frame
  confidence = model.predict(x)[0]
  i = np.argmax(confidence)
  label = classes[i]
  label = "{}: {:,2f}%".format(label,100*confidence[i])
  
  cv2.putText(frm, label, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10,50,70), 1) # Printing the prediction with confidence on frame
  cv2.imshow("Gun Detection", frm) # Printing the label on the frame

  if cv2.waitKey(1) & 0xFF == ord('q'): # creating a command to kill the webcam window
    break
webcam.release()
cv2.destroyAllWindows()