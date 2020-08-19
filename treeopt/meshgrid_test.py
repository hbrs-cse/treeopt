import numpy as np

# x = np.linspace(0., 1., 3)
# y = np.linspace(1., 2., 3)
# z = np.linspace(3., 4., 3)

# xyz = np.dstack(np.meshgrid(x, y, z)).reshape(-1,3)

# print(xyz)

x_step = np.linspace(0, 1, 5)
y_step = np.linspace(0, 1, 5)
z_step = np.linspace(0, 1, 5)
a_step = np.linspace(0, 1, 5)

XY = np.dstack(np.meshgrid(x_step, y_step, z_step, a_step)).reshape(-1, 4)

print(XY)
