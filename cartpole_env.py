import gym
import numpy as np
import pybulletgym
import matplotlib.pyplot as plt
import torch

class InverteddoublependulumEnv():
    def __init__(self):
        self.env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
        self.action_dim = 1
        self.state_dim = 6
    def reset(self):
        state = self.env.reset()
        print(f'state_11: ', state[:-3])
        x, sin_theta1, sin_theta2, cos_theta1, cos_theta2, x_dot, theta1_dot, theta2_dot = state[:-3]
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        state = np.array([x, theta1, theta2, x_dot, theta1_dot, theta2_dot])
        print(f'state_6: ', state)
        return state

    
    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        print(f'state_11: ', state[:-3])
        x, sin_theta1, sin_theta2, cos_theta1, cos_theta2, x_dot, theta1_dot, theta2_dot = state[:-3]
        theta1 = np.arctan2(sin_theta1, cos_theta1)
        theta2 = np.arctan2(sin_theta2, cos_theta2)
        state = np.array([x, theta1, theta2, x_dot, theta1_dot, theta2_dot])
        print(f'state_6: ', state)
        return state, reward, done
    
    def dynamics_analytics(self, state, action):
        # from the source code of the environment
        timestep=0.0165
        state, reward, done = self.step(action)
        return state, reward, done
    def inverted_double_pendulum_dynamics(self, state_batch, u_batch):
        """
        Compute the dynamics of the inverted double pendulum for a batch of states.

        Args:
            state_batch (np.array): Batch of states, shape (batch_size, 6), where each row represents a state vector [x, theta1, theta2, x_dot, theta1_dot, theta2_dot].
            u_batch (np.array): Batch of control inputs, shape (batch_size,), where each element represents a control input for the corresponding state in state_batch.
            L1 (float): Length of the first pendulum.
            L2 (float): Length of the second pendulum.
            m1 (float): Mass of the first pendulum.
            m2 (float): Mass of the second pendulum.
            g (float): Acceleration due to gravity.

        Returns:
            state_dot_batch (np.array): Batch of state derivatives, shape (batch_size, 6), where each row represents the time derivative of the corresponding state in state_batch.
        """
        # physics parameters
        dt = 0.0165
        m_c = 1.0  # 小车质量
        m_1 = 1.0  # 第一根摆杆质量
        m_2 = 1.0  # 第二根摆杆质量
        l_1 = 0.6  # 第一根摆杆长度
        l_2 = 0.6  # 第二根摆杆长度
        g = 9.81 

        # x, theta_1, theta_2, x_dot, theta_dot_1, theta_dot_2 = state
        x = state_batch[:, 0]
        theta_1 = state_batch[:, 1]
        theta_2 = state_batch[:, 2]
        x_dot = state_batch[:, 3]
        theta_dot_1 = state_batch[:, 4]
        theta_dot_2 = state_batch[:, 5]
        u = u_batch[:, 0]

        # 计算一些中间变量
        sin1 = torch.sin(theta_1)
        cos1 = torch.cos(theta_1)
        sin2 = torch.sin(theta_2)
        cos2 = torch.cos(theta_2)

        d11 = m_c + m_1 + m_2
        d12 = m_1 * l_1 * cos1 + m_2 * l_2 * cos2
        d21 = m_1 * l_1 * cos1 + m_2 * l_2 * cos2
        d22 = m_1 * l_1 * l_1 + m_2 * l_2 * l_2 + 2. * m_2 * l_1 * l_2 * torch.cos(theta_1 - theta_2)

        c1 = -m_1 * l_1 * theta_dot_1 * (theta_dot_1 - theta_dot_2) * torch.sin(theta_1 - theta_2) - m_2 * l_1 * theta_dot_1 * theta_dot_1 * torch.sin(theta_1 - theta_2) - m_2 * l_2 * theta_dot_2 * theta_dot_2 * torch.sin(theta_1 - theta_2)
        c2 = m_1 * l_1 * theta_dot_2 * (theta_dot_1 - theta_dot_2) * torch.sin(theta_1 - theta_2) + m_1 * g * torch.sin(theta_1) + m_2 * g * l_2 * torch.sin(theta_2)

        # 计算加速度
        x_dot_dot = (u + d12 * theta_dot_1 * theta_dot_1 * torch.sin(theta_1 - theta_2) - c1) / (m_c + m_1 + m_2 - d21 * d12 / d22)
        theta_1_dot_dot = (u * l_1 * cos1 - c2 * l_1 * cos1 + d22 * theta_dot_1 * theta_dot_1 * torch.sin(theta_1 - theta_2) - (m_c + m_1) * g * l_1 * sin1 + (m_c + m_1) * l_1 * c1) / (d22 * (m_c + m_1) - d12 * d21)
        theta_2_dot_dot = (-u * l_2 * cos2 + c2 * l_2 * cos2 - d21 * theta_dot_2 * theta_dot_2 * torch.sin(theta_1 - theta_2) - m_2 * g * l_2 * sin2 + m_2 * l_2 * c1) / (d22 * (m_c + m_1) - d12 * d21)
        
        x_dot_new = x_dot + x_dot_dot * dt
        theta1_dot_new = theta_dot_1 + theta_1_dot_dot * dt
        theta2_dot_new = theta_dot_2 + theta_2_dot_dot * dt
        x_new = x + x_dot_new * dt
        theta1_new = theta_1 + theta1_dot_new * dt
        theta2_new = theta_2 + theta2_dot_new * dt
        next_state = torch.stack((x_new, theta1_new, theta2_new, x_dot_new, theta1_dot_new, theta2_dot_new), dim=1)
        print(f'next_state: ', next_state.shape)
        return next_state
    
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# env = InverteddoublependulumEnv()
# env.env.seed(42)
# np.random.seed(42)
# env.env.action_space.seed(42)
# env.reset()  
# state = env.step(np.array([0.5]))


# first let's generate a random control sequence
T = 50
control_sequence = np.random.randn(T, 1)
print('control_sequence: ', control_sequence.shape)
# We use the simulator to simulate a trajectory
env = InverteddoublependulumEnv()
start_state = env.reset()
states_pybullet = np.zeros((T+1, 6))
states_pybullet[0] = start_state
for t in range(T):
    states_pybullet[t+1] = env.step(control_sequence[t])[0]
print('states_pybullet: ', states_pybullet.shape)

# Now we will use your analytic dynamics to simulate a trajectory
states_analytic = torch.zeros(T+1, 1, 6) # Need an extra 1 which is the batch dimension (T x B x 4)
states_analytic[0] = torch.from_numpy(start_state).reshape(1, 6)
for t in range(T):
    current_state = states_analytic[t]
    current_control = torch.from_numpy(control_sequence[t]).reshape(1, 1) # add batch dimension to control   
    states_analytic[t+1] = env.inverted_double_pendulum_dynamics(current_state, current_control)
    
# # convert back to numpy for plotting
states_analytic = states_analytic.reshape(T+1, 6).numpy()

# Plot and compare - They should be indistinguishable 
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
axes[0][0].plot(states_analytic[:, 0], label='analytic')
axes[0][0].plot(states_pybullet[:, 0], '--', label='pybullet')
axes[0][0].title.set_text('x')

axes[1][0].plot(states_analytic[:, 1])
axes[1][0].plot(states_pybullet[:, 1], '--')
axes[1][0].title.set_text('theta_1')

axes[2][0].plot(states_analytic[:, 1])
axes[2][0].plot(states_pybullet[:, 2], '--')
axes[2][0].title.set_text('theta_2')

axes[0][1].plot(states_analytic[:, 3])
axes[0][1].plot(states_pybullet[:, 3], '--')
axes[0][1].title.set_text('x_dot')

axes[1][1].plot(states_analytic[:, 4])
axes[1][1].plot(states_pybullet[:, 4], '--')
axes[1][1].title.set_text('theta_dot_1')

axes[2][1].plot(states_analytic[:, 5])
axes[2][1].plot(states_pybullet[:, 5], '--')
axes[2][1].title.set_text('theta_dot_2')

# axes[0][0].legend()
plt.show()