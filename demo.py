import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from numpngw import write_apng
import cv2
import numpy as np

env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
env.seed(42)
np.random.seed(42)
env.action_space.seed(42)

env.reset()
frames = []
frames.append(env.render(mode="rgb_array"))

for i in range(200):
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    print(action)
    img = env.render(mode="rgb_array")
    frames.append(img)
    if done:
        env.reset()

env.close()
write_apng('./InvertedDoublePendulum.png', frames, delay=50)
# show the frames in opencv
for i in range(len(frames)):
    cv2.imshow('frame', frames[i][..., ::-1])
    cv2.waitKey(100)