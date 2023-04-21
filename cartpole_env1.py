import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym
import os

class MyCartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (4,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (6,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        p.resetSimulation()
        # p.setAdditionalSearchPath(pd.getDataPath())
        p.setAdditionalSearchPath(os.path.dirname(os.path.abspath(__file__)))
        self.cartpole = p.loadURDF('cartpole1.urdf')
        # self.cartpole = p.loadURDF('cartpole.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 2, linearDamping=0, angularDamping=0,
                    lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """
        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        theta_1, theta_dot_1 = p.getJointState(self.cartpole, 2)[0:2]
        # it looks like the state is not updated in the simulation
        theta_1 += theta
        theta_dot_1 += theta_dot
        return np.array([x, theta, theta_1, x_dot, theta_dot, theta_dot_1])

    def set_state(self, state):
        x, theta, theta_1, x_dot, theta_dot ,theta_dot_1 = state
        theta_1 -= theta
        theta_dot_1 -= theta_dot
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta, targetVelocity=theta_dot)
        p.resetJointState(self.cartpole, 2, targetValue=theta_1, targetVelocity=theta_dot_1)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        theta_1_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        theta_dot_1_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(low=np.array([x_lims[0], theta_lims[0], theta_1_lims[0], x_dot_lims[0], theta_dot_lims[0], theta_dot_1_lims[0]]),
                                     high=np.array([x_lims[1], theta_lims[1], theta_1_lims[1], x_dot_lims[1], theta_dot_lims[1], theta_dot_1_lims[1]])
                                     )  # linear force # TODO: Verifty that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 240
        self.render_w = 320
        base_pos = [0, 0, 0]
        cam_dist = 2
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=120,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (4,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (4, 4) representing Jacobian df/dx for dynamics f
            B: np.array of shape (4, 1) representing Jacobian df/du for dynamics f
        """
        A, B = None, None
        # --- Your code here
        n = state.shape[0]
        m = control.shape[0]
        A = np.zeros((n, n))
        B = np.zeros((n, m))
        
        # compute A
        # compute B

        for i in range(4):
          for j in range(4):
            delta = np.zeros((4, ))
            delta[j] = 1.
            diff = (self.dynamics(state + eps * delta, control) - self.dynamics(state - eps * delta, control)) / (2 * eps)
            A[i, j] = diff[i]
        
        diff = (self.dynamics(state, control + eps) - self.dynamics(state, control - eps)) / (2 * eps)
        B = diff.reshape((4, 1))
        # ---
        return A, B
    
def dynamics_analytic_batch(state, action):
    next_state = None
    dt = 0.05
    g = 9.81
    m0 = 1.0
    m1 = 0.1
    m2 = 0.1
    L1 = 1.0
    L2 = 1.0
    
    action = -action
    
    B = state.shape[0]
    state_dim = state.shape[1]
    action_dim = action.shape[1]
    x_dot_dot = torch.zeros((B, 1))
    theta_1_dot_dot = torch.zeros((B, 1))
    theta_2_dot_dot = torch.zeros((B, 1))

    x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
    u = action

    # construct D
    D = torch.zeros((B, 3, 3))
    D[:, 0, 0] = m0 + m1 + m2
    D[:, 0, 1] = (0.5 * m1 + m2) * L1 * torch.cos(theta_1)
    D[:, 0, 2] = 0.5 * m2 * L2 * torch.cos(theta_2)
    D[:, 1, 0] = D[:, 0, 1]
    D[:, 1, 1] = (1.0 / 3.0 * m1 + m2) * L1 * L1
    D[:, 1, 2] = 0.5 * m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
    D[:, 2, 0] = D[:, 0, 2]
    D[:, 2, 1] = D[:, 1, 2]
    D[:, 2, 2] = 1.0 / 3.0 * m2 * L2 * L2

    # construct C
    C = torch.zeros((B, 3, 3))
    C[:, 0, 1] = -(0.5 * m1 + m2) * L1 * torch.sin(theta_1) * theta_1_dot
    C[:, 0, 2] = -0.5 * m2 * L2 * torch.sin(theta_2) * theta_2_dot
    C[:, 1, 2] = 0.5 * m2 * L1 * L2 * torch.sin(theta_1 - theta_2) * theta_2_dot
    C[:, 2, 1] = -0.5 * m2 * L1 * L2 * torch.sin(theta_1 - theta_2) * theta_1_dot

    # construct G
    G = torch.zeros((B, 3, 1))
    G[:, 1, 0] = -0.5 * (m1 + m2) * g * L1 * torch.sin(theta_1)
    G[:, 2, 0] = -0.5 * m2 * g * L2 * torch.sin(theta_2)

    # construct H
    H = torch.zeros((B, 3, 1))
    H[:, 0, 0] = 1

    # compute x_dot_dot, theta_1_dot_dot, theta_2_dot_dot
    state_dot = torch.cat((x_dot, theta_1_dot, theta_2_dot)).view(B, 3, 1)
    
    # temp shape 100x3x1
    temp = torch.bmm(-torch.inverse(D), (torch.bmm(C, state_dot) + G + torch.bmm(H, u.view(B, 1, 1))))
    
    x_dot_dot = temp[:, 0, 0].squeeze()
    theta_1_dot_dot = temp[:, 1, 0].squeeze()
    theta_2_dot_dot = temp[:, 2, 0].squeeze()
    
    x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
    x_dot_new = x_dot + dt * x_dot_dot
    theta_1_dot_new = theta_1_dot + dt * theta_1_dot_dot
    theta_2_dot_new = theta_2_dot + dt * theta_2_dot_dot
    x_new = x + dt * x_dot_new
    theta_1_new = theta_1 + dt * theta_1_dot_new
    theta_2_new = theta_2 + dt * theta_2_dot_new
    next_state = torch.stack((x_new, theta_1_new, theta_2_new, x_dot_new, theta_1_dot_new, theta_2_dot_new), dim=1)
    # ---
    return next_state

def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    m0 = 1.0
    m1 = 0.1
    m2 = 0.1
    L1 = 1.0
    L2 = 1.0

    action = -action
    # --- Your code here
    # for-loop version
    B = state.shape[0]
    state_dim = state.shape[1]
    action_dim = action.shape[1]
    x_dot_dot = torch.zeros((B, ))
    theta_1_dot_dot = torch.zeros((B, ))
    theta_2_dot_dot = torch.zeros((B, ))
    for b in range(B):
        x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot = state[b, :]
        u = action[b, :]
        # construct D
        D = torch.zeros((3, 3))
        D[0, 0] = m0 + m1 + m2
        D[0, 1] = (0.5 * m1 + m2) * L1 * torch.cos(theta_1)
        D[0, 2] = 0.5 * m2 * L2 * torch.cos(theta_2)
        D[1, 0] = D[0, 1]
        D[1, 1] = (1.0 / 3.0 * m1 + m2) * L1 * L1
        D[1, 2] = 0.5 * m2 * L1 * L2 * torch.cos(theta_1 - theta_2)
        D[2, 0] = D[0, 2]
        D[2, 1] = D[1, 2]
        D[2, 2] = 1.0 / 3.0 * m2 * L2 * L2
        # construct C
        C = torch.zeros((3, 3))
        C[0, 1] = -(0.5 * m1 + m2) * L1 * torch.sin(theta_1) * theta_1_dot
        C[0, 2] = -0.5 * m2 * L2 * torch.sin(theta_2) * theta_2_dot
        C[1, 2] = 0.5 * m2 * L1 * L2 * torch.sin(theta_1 - theta_2) * theta_2_dot
        C[2, 1] = -0.5 * m2 * L1 * L2 * torch.sin(theta_1 - theta_2) * theta_1_dot
        # construct G
        G = torch.zeros((3, 1))
        G[1, 0] = -0.5 * (m1 + m2) * g * L1 * torch.sin(theta_1)
        G[2, 0] = -0.5 * m2 * g * L2 * torch.sin(theta_2)
        # construct H
        H = torch.zeros((3, 1))
        H[0, 0] = 1

        # print("x: ", x)
        # print("theta_1: ", theta_1)
        # print("theta_2: ", theta_2)
        
        # compute x_dot_dot, theta_1_dot_dot, theta_2_dot_dot
        temp = -torch.inverse(D) @ (C @ torch.tensor([[x_dot], [theta_1_dot], [theta_2_dot]]) + G + H * u)
        x_dot_dot[b] = temp[0, 0]
        theta_1_dot_dot[b] = temp[1, 0]
        theta_2_dot_dot[b] = temp[2, 0]
    # print("x_dot_dot: ", x_dot_dot)
    # print("theta_1_dot_dot: ", theta_1_dot_dot)
    # print("theta_2_dot_dot: ", theta_2_dot_dot)

    x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
    x_dot_new = x_dot + dt * x_dot_dot
    theta_1_dot_new = theta_1_dot + dt * theta_1_dot_dot
    theta_2_dot_new = theta_2_dot + dt * theta_2_dot_dot
    x_new = x + dt * x_dot_new
    theta_1_new = theta_1 + dt * theta_1_dot_new
    theta_2_new = theta_2 + dt * theta_2_dot_new
    # theta_dot_new = theta_dot + dt * theta_dot_dot
    # x_dot_new = x_dot + dt * x_dot_dot
    # theta_new = theta + dt * theta_dot_new
    # x_new = x + dt * x_dot_new

    next_state = torch.stack((x_new, theta_1_new, theta_2_new, x_dot_new, theta_1_dot_new, theta_2_dot_new), 1)
    # ---

    return next_state


def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here

    # A = torch.autograd.functional.jacobian(dynamics_analytic, state)
    # B = torch.autograd.functional.jacobian(dynamics_analytic, control)
    # A, B = torch.autograd.functional.jacobian(dynamics_analytic, (state.reshape((1, 4)), control.reshape(1, 1)))
    A = torch.autograd.functional.jacobian(lambda x : dynamics_analytic(x, control.reshape((1, 1))), state.reshape((1, 4)))
    B = torch.autograd.functional.jacobian(lambda x : dynamics_analytic(state.reshape((1, 4)), x), control.reshape((1, 1)))
    A = A.reshape((4, 4))
    B = B.reshape((4, 1))
    # ---
    return A, B

def minus_pi_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi