import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable


class VisualizeMetamodel:
    """
    Class that contains functions to visualize a metamodel. At least the metamodel as well as the boundaries of the
    visualization have to be parametrizes in its designated setter functions. By executing the plot_metamodel function,
    an matplotlib plot is generated, where the plane of observation can be selected by setting radio buttons and
    sliders.
    """
    def __init__(self):
        self._metamodel = None
        self._bounds = None
        self._points_per_axis = 100
        self._current_values = []
        self._x_axis_num = 0
        self._y_axis_num = 1
        self._c_levels = 100
        self._eq_lines = 10
        self._num_doe_points = None

    @property
    def metamodel(self):
        return self._metamodel

    @metamodel.setter
    def metamodel(self, model):
        self._metamodel = model

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, array):
        self._bounds = array

    @property
    def points_per_axis(self):
        return self._points_per_axis

    @points_per_axis.setter
    def points_per_axis(self, number):
        self._points_per_axis = number

    @property
    def current_values(self):
        return self._current_values

    @current_values.setter
    def current_values(self, values):
        self._current_values = values

    @property
    def x_axis_num(self):
        return self._x_axis_num

    @x_axis_num.setter
    def x_axis_num(self, number):
        self._x_axis_num = number

    @property
    def y_axis_num(self):
        return self._y_axis_num

    @y_axis_num.setter
    def y_axis_num(self, number):
        self._y_axis_num = number

    @property
    def c_levels(self):
        return self._c_levels

    @c_levels.setter
    def c_levels(self, number):
        self._c_levels = number

    @property
    def eq_lines(self):
        return self._eq_lines

    @eq_lines.setter
    def eq_lines(self, number):
        self._eq_lines = number

    @property
    def num_doe_points(self):
        return self._num_doe_points

    @num_doe_points.setter
    def num_doe_points(self, number):
        self._num_doe_points = number

    def plot_metamodel(self):
        """
        Creates a plot showing one plane of the metamodel. The plane can be changed by selecting radio buttons. The
        other parameters are constant and can be set via sliders on the plot. 
        """
        fig, ax = plt.subplots()

        plt.subplots_adjust(left=0.15)
        plt.subplots_adjust(bottom=0.1 + 0.05 * self.bounds.shape[0])

        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")

        linspaces = []
        for bound in self.bounds:
            linspaces.append(np.linspace(bound[0], bound[1], self.points_per_axis))
            self.current_values.append(np.average(bound))

        def replot():
            plot_points = [None] * self.bounds.shape[0]
            for i, bound in enumerate(self.bounds):
                if i == min(self.x_axis_num, self.y_axis_num):
                    x_coords, y_coords = np.meshgrid(linspaces[self.x_axis_num], linspaces[self.y_axis_num])

                    plot_points[self.x_axis_num] = x_coords.reshape(-1)
                    plot_points[self.y_axis_num] = y_coords.reshape(-1)

                elif i == max(self.x_axis_num, self.y_axis_num):
                    continue

                else:
                    plot_points[i] = self.current_values[i] * np.ones(
                        [self.points_per_axis, self.points_per_axis]).reshape(-1)

            plot_points = np.array(plot_points).T

            sm_result = self.metamodel.predict_values(plot_points)

            ax.clear()

            im = ax.contourf(plot_points[:, self.x_axis_num].reshape(self.points_per_axis, self.points_per_axis),
                            plot_points[:, self.y_axis_num].reshape(self.points_per_axis, self.points_per_axis),
                            sm_result.reshape(self.points_per_axis, self.points_per_axis), self.c_levels)

            ax.contour(plot_points[:, self.x_axis_num].reshape(self.points_per_axis, self.points_per_axis),
                       plot_points[:, self.y_axis_num].reshape(self.points_per_axis, self.points_per_axis),
                       sm_result.reshape(self.points_per_axis, self.points_per_axis), self.eq_lines, colors="black")

            images = []

            if False:
                images.append(ax.plot(self.metamodel.training_points[None][0][0][self.num_doe_points:][:,self.x_axis_num],
                        self.metamodel.training_points[None][0][0][self.num_doe_points:][:,self.y_axis_num], "rx"))

                images.append(ax.plot(self.metamodel.training_points[None][0][0][:self.num_doe_points][:,self.x_axis_num],
                        self.metamodel.training_points[None][0][0][:self.num_doe_points][:,self.y_axis_num], "bx"))

                y_opt = min(self.metamodel.training_points[None][0][1])
                x_opt = self.metamodel.training_points[None][0][0][self.metamodel.training_points[None][0][1].tolist().index(y_opt)]

                images.append(ax.plot(x_opt[self.x_axis_num], x_opt[self.y_axis_num], "gx"))


            #images.append(ax.plot(self.))

            # images.append(ax.grid())

            cax.clear()

            fig.colorbar(im, cax=cax)

            plt.show()

        def update(val):
            current_values = []
            for slider in sliders:
                current_values.append(slider.val)

            self.current_values = current_values

            replot()

        sliders = []
        for i, bound in enumerate(self.bounds):
            sliders.append(
                Slider(
                    ax=plt.axes([0.25, 0.05 + 0.05 * i, 0.65, 0.03]),
                    label="parameter_" + str(i),
                    valmin=bound[0],
                    valmax=bound[1],
                    valinit=np.average(bound)
                )
            )
            sliders[-1].on_changed(update)

        axis_numbers = list(range(0, self.bounds.shape[0]))
        axis_numbers_str = ["parameter_" + str(i) for i in axis_numbers]
        axis_dict = dict(zip(axis_numbers_str, axis_numbers))

        rax = plt.axes([0.05, 0.7, 0.075, 0.15])
        x_radio = RadioButtons(rax, tuple(axis_numbers_str))

        def x_axis(label):
            self.x_axis_num = axis_dict[label]
            replot()

        x_radio.on_clicked(x_axis)

        rax = plt.axes([0.05, 0.4, 0.075, 0.15])
        y_radio = RadioButtons(rax, tuple(axis_numbers_str))

        def y_axis(label):
            self.y_axis_num = axis_dict[label]
            replot()

        y_radio.on_clicked(y_axis)
        y_radio.set_active(self.y_axis_num)

        plt.show()

if __name__ == "__main__":
    import smt.sampling_methods as sampling
    import smt.surrogate_models as smt

    def get_sm(fun, bounds, doe_num):
        doe = sampling.LHS(xlimits=bounds)
        x_0 = doe(doe_num)
        y_0 = fun(*tuple([x_0[:, i] for i in range(bounds.shape[0])]))

        sm = smt.RBF(d0=5)
        sm.set_training_values(x_0, y_0)
        sm.train()

        return (sm)


    def fun(x, y, z, a):
        return x ** 2 + y ** 2 + (x * z) ** 2 + y+ z + np.sin(a)


    bounds = np.array([[-1, 1], [0, 2], [2, 6], [0,50]])
    sm = get_sm(fun, bounds, 50)

    vis = VisualizeMetamodel()
    vis.bounds = bounds
    vis.metamodel = sm
    vis.num_doe_points = 25
    vis.points_per_axis = 100

    vis.plot_metamodel()
