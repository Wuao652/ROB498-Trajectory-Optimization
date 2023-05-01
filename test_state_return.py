import numpy as np
import matplotlib.pyplot as plt
from cartpole_env1 import *

np.set_printoptions(precision=3, suppress=True)
myenv = MyCartpoleEnv()
myenv.reset(state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

# frames to create animated png
frames=[] 
img = myenv.render().astype(np.uint8)
frames.append(img)

# use plt to visualize the img 
plt.figure()
plt.imshow(img)
plt.pause(0.01)
plt.clf()

for i in range(100):
    action = np.array([1.0])
    s = myenv.step(action)
    img = myenv.render().astype(np.uint8)
    frames.append(img)
    print(f"===================={i}====================")
    print("state: ", s)
    plt.title(f'Action control, state: {s},')
    # plt.axis('off')
    plt.imshow(img)
    plt.pause(0.01)
    plt.clf()