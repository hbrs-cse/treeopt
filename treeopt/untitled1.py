from treeopt.ego import OptimizeEGO
from treeopt.visualize2 import VisualizeMetamodel
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from smt.sampling_methods import FullFactorial

from smt.surrogate_models import KRG


def problem(x):
    f_x = np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4

    return f_x
    #return 0.25 * x[0]**0.5
    #return np.array((x[0]-1)**2-1)

ego = OptimizeEGO()
ego.opti_object = problem
ego.x_limits = np.array([[0.5, 2.5]])
sampling = FullFactorial(xlimits = ego.x_limits)
ego.doe_data = sampling(3)

ego.do_ego(0.0078125)

print(ego.num_evals, ego.x_opt, ego.y_opt)

def problem_plot(x):
    f_x = np.sin(10 * np.pi * x) / (2 * x) + (x - 1) ** 4

    return f_x

sm = KRG(theta0=[1e-2])
sm.set_training_values(np.array(ego.x_data), np.array(ego.y_data))
sm.train()

num = 100
x = np.linspace(0.5, 2.5, num)

y = sm.predict_values(x)
y_org = problem_plot(x)


fig, axs = plt.subplots(1)

# add a plot with variance
axs.plot(np.array(ego.x_data), np.array(ego.y_data), "rx")
axs.plot(x, y)
axs.plot(x, y_org, "g-")

axs.set_xlabel("x")
axs.set_ylabel("y")
plt.xlim([0.5, 2.5])
axs.legend(
    ["Training data", "Metamodell", "Funktion"],
    loc="lower right",
)
plt.grid()
plt.show()