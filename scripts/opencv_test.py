"""opencv test"""

import numpy as np;
import cv2
import skvideo.io as skv;
import matplotlib.pyplot as plt;

cap = skv.VideoCapture("/home/arlmonster/workspace/telaugesa/data/000046280.avi")
 
  
while(cap.isOpened()):
    ret, frame = cap.read()
     
    if frame is None:
        break;
     
    frame=np.asarray(frame);
     
    plt.figure(1);
    plt.imshow(frame);
    plt.show();
     
    print "hello"
