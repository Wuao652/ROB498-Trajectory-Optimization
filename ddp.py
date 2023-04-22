import gym
from autograd import grad, jacobian
import autograd.numpy as np
from cartpole_env1 import MyCartpoleEnv
# from cartpole_env1 import dynamics_analytic_np
import matplotlib.pyplot as plt



def dynamics_analytic_np(state, action):
    """
        Computes next state of cartpole given current state and action using analytic dynamics

    Args:
        state: numpy array of shape (6,) representing cartpole state
        action: numpy array of shape (1,) representing the force to apply

    Returns:
        next_state: numpy array of shape (6,) representing next cartpole state
    """
    dt = 0.05
    g = 9.81
    m0 = 1.0
    m1 = 0.1
    m2 = 0.1
    L1 = 1.0
    L2 = 1.0

    next_state = None
    x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot = state
    u = action[0]
    # construct D
    D = [m0 + m1 + m2, (0.5 * m1 + m2) * L1 * np.cos(theta_1), 0.5 * m2 * L2 * np.cos(theta_2),
         (0.5 * m1 + m2) * L1 * np.cos(theta_1), (m1 / 3.0 + m2) * L1 ** 2, 0.5 * m2 * L1 * L2 * np.cos(theta_1 - theta_2),
         0.5 * m2 * L2 * np.cos(theta_2), 0.5 * m2 * L1 * L2 * np.cos(theta_1 - theta_2), (m2 / 3.0) * L2 ** 2]
    D = np.stack(D, axis=0).reshape(3, 3)

    # construct C
    C = [0, -(0.5 * m1 + m2) * L1 * np.sin(theta_1) * theta_1_dot, -0.5 * m2 * L2 * np.sin(theta_2) * theta_2_dot,
         0, 0, 0.5 * m2 * L1 * L2 * np.sin(theta_1 - theta_2) * theta_2_dot,
         0, -0.5 * m2 * L1 * L2 * np.sin(theta_1 - theta_2) * theta_1_dot, 0]
    C = np.stack(C, axis=0).reshape(3, 3)

    # construct G
    G = [0, -0.5 * (m1 + m2) * g * L1 * np.sin(theta_1), -0.5 * m2 * g * L2 * np.sin(theta_2)]
    G = np.stack(G, axis=0).reshape(3, 1)

    # construct H
    H = np.zeros((3, 1))
    H[0, 0] = 1.0
    
    temp = -np.linalg.inv(D) @ (C @ np.stack([x_dot, theta_1_dot, theta_2_dot], axis=0).reshape(3, 1) + G - H * u)

    x_dot_dot, theta_1_dot_dot, theta_2_dot_dot = temp.reshape(-1) 
    
    # compute next state
    x_dot_new = x_dot + dt * x_dot_dot
    theta_1_dot_new = theta_1_dot + dt * theta_1_dot_dot
    theta_2_dot_new = theta_2_dot + dt * theta_2_dot_dot
    x_new = x + dt * x_dot_new
    theta_1_new = theta_1 + dt * theta_1_dot_new
    theta_2_new = theta_2 + dt * theta_2_dot_new
    next_state = np.stack([x_new, theta_1_new, theta_2_new, x_dot_new, theta_1_dot_new, theta_2_dot_new], axis=0).astype(np.float32)
    # next_state = np.array([x_new, theta_1_new, theta_2_new, x_dot_new, theta_1_dot_new, theta_2_dot_new]).astype(np.float32)
    return next_state


class DDP:
    def __init__(self, next_state, running_cost, final_cost,
                 umax, state_dim, pred_time=50):
        self.pred_time = pred_time
        self.umax = umax
        self.v = [0.0 for _ in range(pred_time + 1)]
        self.v_x = [np.zeros(state_dim) for _ in range(pred_time + 1)]
        self.v_xx = [np.zeros((state_dim, state_dim)) for _ in range(pred_time + 1)]
        self.f = next_state
        self.lf = final_cost
        self.lf_x = grad(self.lf)
        self.lf_xx = jacobian(self.lf_x)
        self.l_x = grad(running_cost, 0)
        self.l_u = grad(running_cost, 1)
        self.l_xx = jacobian(self.l_x, 0)
        self.l_uu = jacobian(self.l_u, 1)
        self.l_ux = jacobian(self.l_u, 0)
        self.f_x = jacobian(self.f, 0)
        self.f_u = jacobian(self.f, 1)
        self.f_xx = jacobian(self.f_x, 0)
        self.f_uu = jacobian(self.f_u, 1)
        self.f_ux = jacobian(self.f_u, 0)

    def backward(self, x_seq, u_seq):
        self.v[-1] = self.lf(x_seq[-1])
        self.v_x[-1] = self.lf_x(x_seq[-1])
        self.v_xx[-1] = self.lf_xx(x_seq[-1])
        k_seq = []
        kk_seq = []
        for t in range(self.pred_time - 1, -1, -1):
            f_x_t = self.f_x(x_seq[t], u_seq[t])
            f_u_t = self.f_u(x_seq[t], u_seq[t])
            q_x = self.l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, self.v_x[t + 1])
            q_u = self.l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, self.v_x[t + 1])
            q_xx = self.l_xx(x_seq[t], u_seq[t]) + \
              np.matmul(np.matmul(f_x_t.T, self.v_xx[t + 1]), f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_xx(x_seq[t], u_seq[t])))
            tmp = np.matmul(f_u_t.T, self.v_xx[t + 1])
            q_uu = self.l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_uu(x_seq[t], u_seq[t])))
            q_ux = self.l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) + \
              np.dot(self.v_x[t + 1], np.squeeze(self.f_ux(x_seq[t], u_seq[t])))
            inv_q_uu = np.linalg.inv(q_uu)
            k = -np.matmul(inv_q_uu, q_u)
            kk = -np.matmul(inv_q_uu, q_ux)
            dv = 0.5 * np.matmul(q_u, k)
            self.v[t] += dv
            self.v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            self.v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
            k_seq.append(k)
            kk_seq.append(kk)
        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq

    def forward(self, x_seq, u_seq, k_seq, kk_seq):
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        for t in range(len(u_seq)):
            control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, -self.umax, self.umax)
            x_seq_hat[t + 1] = self.f(x_seq_hat[t], u_seq_hat[t])
        return x_seq_hat, u_seq_hat

# set deterministic seed
np.random.seed(42)
env = MyCartpoleEnv()
env.reset(state=np.array([0., 0.1, 0.1, 0., 0., 0.]))

state_0 = env.get_state()
state = state_0
frames = []
frames.append(env.render())


ddp = DDP(next_state=dynamics_analytic_np,  # x(i+1) = f(x(i), u)
          running_cost=lambda x, u: 0.5 * np.sum(np.square(u)),  # l(x, u)
          final_cost=lambda x: (np.square(x[0]) + np.square(x[1]) + np.square(x[2])),  # lf(x)
          umax = 30.0,
          state_dim = 6,
          pred_time=10)
u_seq = [np.zeros(1) for _ in range(ddp.pred_time)]
x_seq = [state.copy()]
for t in range(ddp.pred_time):
    x_seq.append(dynamics_analytic_np(x_seq[-1], u_seq[t]))

cnt = 0
while True:
    #import pyglet
    #pyglet.image.get_buffer_manager().get_color_buffer().save('frame_%04d.png' % cnt)
    for _ in range(2):
        k_seq, kk_seq = ddp.backward(x_seq, u_seq)
        x_seq, u_seq = ddp.forward(x_seq, u_seq, k_seq, kk_seq)

    # print(u_seq.T)
    print("=====================================")
    state = env.step(u_seq[0])
    x_seq[0] = state.copy()
    cnt += 1
    frame = env.render()
    frames.append(frame)

    plt.title(f'DDP control')
    # plt.axis('off')
    plt.imshow(frame)
    plt.pause(0.01)
    plt.clf()

