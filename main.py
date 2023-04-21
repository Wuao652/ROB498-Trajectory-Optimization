import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from numpngw import write_apng
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

from mppi import MPPI
from cartpole_env1 import MyCartpoleEnv
from cartpole_env1 import dynamics_analytic

def swing_up_cost_function(state, action):
    """
    Compute the state cost for MPPI.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    target_pose = target_pose.reshape((1,6))
    Q = torch.tensor([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0.1, 0., 0.],
                      [0., 0., 0., 0., 0.1, 0.],
                      [0., 0., 0., 0., 0., 0.1]]
    )
    delta = state - target_pose #[B, 3]
    cost = delta @ Q @ delta.T
    cost = torch.diag(cost, 0)

    # ---
    return cost

class MPPI_Controller(object):
    """
    MPPI-based controller for the InvertedDoublePendulum environment.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        # self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = 6
        action_dim = 1
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        A batched version of the dynamics model using the provided state and action tensors.
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states.
        """
        next_state = None
        # --- Your code here
        next_state = dynamics_analytic(state, action)
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        print(state)
        print(state.dtype)
        state_tensor = torch.tensor(state)
        action_tensor = self.mppi.command(state_tensor)
        action = action_tensor.detach().cpu().numpy()
        # ---
        return action
    def set_target_state(self, state):
        self.target_state = state
    
env = MyCartpoleEnv()
np.random.seed(42)
env.reset(state=np.array([0., 0., 0., 0., 0., 0.]))
state_0 = env.get_state()
state = state_0
frames = []
frames.append(env.render())

controller = MPPI_Controller(env=env, model=None, cost_function=swing_up_cost_function,num_samples=10, horizon=10)
TARGET_POSE_FREE_TENSOR = torch.tensor([0., 0., 0., 0., 0., 0.])

for i in range(20):
    action = controller.control(state)
    state = env.step(action)
    print(action)
    frame = env.render()
    frames.append(frame)

for i in range(len(frames)):
    cv2.imshow('frame', frames[i][..., ::-1])
    cv2.waitKey(0)


