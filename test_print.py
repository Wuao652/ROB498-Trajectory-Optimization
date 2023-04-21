import matplotlib.pyplot as plt
from numpngw import write_apng
from cartpole_env1 import *

import numpy as np

# myenv.reset(state = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]))

# myframe = myenv.render()
# plt.imshow(myframe)
# plt.title("new env")
# plt.show()

# frames=[] #frames to create animated png
# frames.append(myenv.render())
# for i in range(100):
#     action = np.array([1.0])
#     s = myenv.step(action)
#     img = myenv.render()
#     frames.append(img)

# write_apng("cartpole_example.png", frames, delay=10)
# # Image(filename="cartpole_example.png")
## Let's test to see if your analytic dynamics matches the simulator



# set deterministic seed
np.random.seed(2)

# first let's generate a random control sequence
T = 10
control_sequence = np.random.randn(T, 1).astype(np.float32)
start_state = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
env = MyCartpoleEnv()
env.reset(start_state)

# We use the simulator to simulate a trajectory
states_pybullet = np.zeros((T+1, 6))
states_pybullet[0] = start_state
for t in range(T):
    states_pybullet[t+1] = env.step(control_sequence[t])
print('states_pybullet: ', states_pybullet.shape)

current_state = torch.from_numpy(start_state).reshape(1, 6)
current_control = torch.from_numpy(control_sequence[0]).reshape(1, 1) # add batch dimension to control
dynamics_analytic(current_state, current_control)
print("good")
# Now we will use your analytic dynamics to simulate a trajectory
states_analytic = torch.zeros(T+1, 1, 6) # Need an extra 1 which is the batch dimension (T x B x 4)
states_analytic[0] = torch.from_numpy(start_state).reshape(1, 6)
for t in range(T):
    current_state = states_analytic[t]
    current_control = torch.from_numpy(control_sequence[t]).reshape(1, 1) # add batch dimension to control    
    states_analytic[t+1] = dynamics_analytic(current_state, current_control)
    
# convert back to numpy for plotting
states_analytic = states_analytic.reshape(T+1, 6).numpy()

states_analytic = -states_analytic

# Plot and compare - They should be indistinguishable 
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
axes[0][0].plot(states_analytic[:, 0], label='analytic')
axes[0][0].plot(states_pybullet[:, 0], '--', label='pybullet')
axes[0][0].title.set_text('x')

axes[1][0].plot(states_analytic[:, 1])
axes[1][0].plot(states_pybullet[:, 1], '--')
axes[1][0].title.set_text('theta')

axes[2][0].plot(states_analytic[:, 2])
axes[2][0].plot(states_pybullet[:, 2], '--')
axes[2][0].title.set_text('theta_1')

axes[0][1].plot(states_analytic[:, 3])
axes[0][1].plot(states_pybullet[:, 3], '--')
axes[0][1].title.set_text('x_dot')

axes[1][1].plot(states_analytic[:, 4])
axes[1][1].plot(states_pybullet[:, 4], '--')
axes[1][1].title.set_text('theta_dot')

axes[2][1].plot(states_analytic[:, 5])
axes[2][1].plot(states_pybullet[:, 5], '--')
axes[2][1].title.set_text('theta_1_dot')

axes[0][0].legend()
plt.show()
