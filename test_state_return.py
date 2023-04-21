import matplotlib.pyplot as plt
from numpngw import write_apng
from cartpole_env1 import *

import numpy as np
import cv2

np.set_printoptions(precision=3, suppress=True)
myenv = MyCartpoleEnv()
myenv.reset(state = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]))

frames=[] #frames to create animated png
img = myenv.render()
frames.append(img)

# use cv2 to display the image
cv2.imshow('image', img)
cv2.waitKey(0)
for i in range(100):
    action = np.array([1.0])
    s = myenv.step(action)
    img = myenv.render()
    frames.append(img)
    cv2.imshow('image', img[...,::-1])
    print(f"===================={i}====================")
    print("state: ", s)
    cv2.waitKey(0)
    # use cv2 to print the state
    
