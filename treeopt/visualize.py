import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import numpy as np


class Visualize:
    def __init__(self, optimizer):
        """
        Initializes a visualize Class

        :param optimizer: Object of the TreeOpt Class containing all
            information of the calculated metamodel
        :type optimizer: TreeOpt Class Object
        """

        self.opti_data = optimizer
        self.num_ele = 100

    def set_number_of_points(self, n):
        """
        Sets the number of points which on which a metamodel is to be
        evaluated alongside each axis, in order to generate a plot

        :param n: Number of Evaluations alongside each axis (default n=100)
        :type n: Integer
        :return: Nothing
        :rtype: None
        """
        self.num_ele = n

    def plot_benchmark_2vars(self):
        """
        Plots a benchmarking function on the right and the calculated
        metamodel on the left Outputs a matplotlib graph
        """

        x_step = np.linspace(
            self.opti_data.limits[0, 0],
            self.opti_data.limits[0, 1],
            self.num_ele,
        )
        y_step = np.linspace(
            self.opti_data.limits[1, 0],
            self.opti_data.limits[1, 1],
            self.num_ele,
        )
        XY = np.dstack(np.meshgrid(x_step, y_step)).reshape(-1, 2)

        fun1Vars = self.opti_data.execute_problem(XY.astype(np.float)).reshape(
            self.num_ele, self.num_ele
        )
        fun2Vars = self.opti_data.sm.predict_values(
            XY.astype(np.float)
        ).reshape(self.num_ele, self.num_ele)

        eqlines = 10
        cfsteps = 100
        v_min = np.floor(np.min([fun1Vars.reshape(-1)])).astype(np.float)
        v_max = np.ceil(np.max([fun1Vars.reshape(-1)])).astype(np.float)
        fig, axes = plt.subplots(nrows=1, ncols=2)

        images = []

        images.append(
            axes[0].contourf(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun1Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes[0].contour(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun1Vars,
                eqlines,
                colors="black",
            )
        )

        images.append(
            axes[1].contourf(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun2Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes[1].contour(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun2Vars,
                eqlines,
                colors="black",
            )
        )

        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        for im in images:
            im.set_norm(norm)

        for i in range(axes.shape[0]):
            images.append(
                axes[i].scatter(
                    np.atleast_2d(self.opti_data.x[: self.opti_data.numDOE])[
                        :, 0
                    ].reshape(-1),
                    np.atleast_2d(self.opti_data.x[: self.opti_data.numDOE])[
                        :, 1
                    ].reshape(-1),
                    marker="x",
                    color="r",
                    label="Points-DOE",
                )
            )
            images.append(
                axes[i].scatter(
                    np.atleast_2d(self.opti_data.x[self.opti_data.numDOE :])[
                        :, 0
                    ].reshape(-1),
                    np.atleast_2d(self.opti_data.x[self.opti_data.numDOE :])[
                        :, 1
                    ].reshape(-1),
                    marker="x",
                    color="m",
                    label="Expanded Points",
                )
            )
            images.append(
                axes[i].scatter(
                    np.atleast_2d(self.opti_data.current_best_point[0]),
                    self.opti_data.current_best_point[1],
                    marker="x",
                    color="white",
                )
            )

        images.append(axes[0].set_title("Benchmarking function"))
        images.append(axes[1].set_title("Calculated metamodell"))

        fig.subplots_adjust(bottom=0.225, top=0.85)

        axes = np.hstack([axes, fig.add_axes([0.125, 0.075, 0.775, 0.075])])

        images.append(
            mpl.colorbar.ColorbarBase(
                axes[-1],
                cmap=mpl.cm.viridis,
                norm=norm,
                orientation="horizontal",
            )
        )

        if hasattr(self.opti_data, "name"):
            fig.suptitle(self.opti_data.name)

        lines = []
        labels = []

        axLine, axLabel = axes[0].get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

        fig.legend(
            lines,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
        )

        plt.show()

    def plot_metamodell_1vars(self):
        """
        Plots a metamodel with two one input variable and one output variable
        """

        x_step = np.linspace(
            self.opti_data.limits[0, 0],
            self.opti_data.limits[0, 1],
            self.num_ele,
        )

        fun1Vars = self.opti_data.sm.predict_values(x_step)

        fig, axes = plt.subplots(nrows=1, ncols=1)

        images = []

        images.append(plt.plot(x_step, fun1Vars, "-", c="blue"))
        images.append(plt.plot(self.opti_data.x, self.opti_data.y, "x", c="r"))
        images.append(plt.grid())

        if hasattr(self.opti_data, "name"):
            fig.suptitle(self.opti_data.name)

        plt.show()

    def plot_metamodell_2vars(self):
        """
        Plots a Metamodel with two input variables and one output variable
        """

        x_step = np.linspace(
            self.opti_data.limits[0, 0],
            self.opti_data.limits[0, 1],
            self.num_ele,
        )
        y_step = np.linspace(
            self.opti_data.limits[1, 0],
            self.opti_data.limits[1, 1],
            self.num_ele,
        )
        XY = np.dstack(np.meshgrid(x_step, y_step)).reshape(-1, 2)

        fun1Vars = self.opti_data.sm.predict_values(
            XY.astype(np.float)
        ).reshape(self.num_ele, self.num_ele)

        eqlines = 10
        cfsteps = 100
        v_min = np.floor(np.min([fun1Vars.reshape(-1)])).astype(np.float)
        v_max = np.ceil(np.max([fun1Vars.reshape(-1)])).astype(np.float)
        fig, axes = plt.subplots(nrows=1, ncols=1)

        images = []

        images.append(
            axes.contourf(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun1Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes.contour(
                XY[:, 0].reshape(self.num_ele, self.num_ele),
                XY[:, 1].reshape(self.num_ele, self.num_ele),
                fun1Vars,
                eqlines,
                colors="black",
            )
        )

        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        for im in images:
            im.set_norm(norm)

        images.append(
            axes.scatter(
                np.atleast_2d(self.opti_data.x[: self.opti_data.numDOE])[
                    :, 0
                ].reshape(-1),
                np.atleast_2d(self.opti_data.x[: self.opti_data.numDOE])[
                    :, 1
                ].reshape(-1),
                marker="x",
                color="r",
            )
        )
        images.append(
            axes.scatter(
                np.atleast_2d(self.opti_data.x[self.opti_data.numDOE :])[
                    :, 0
                ].reshape(-1),
                np.atleast_2d(self.opti_data.x[self.opti_data.numDOE :])[
                    :, 1
                ].reshape(-1),
                marker="x",
                color="m",
            )
        )

        fig.subplots_adjust(bottom=0.225, top=0.85)

        ax2 = fig.add_axes([0.125, 0.075, 0.775, 0.075])

        images.append(
            mpl.colorbar.ColorbarBase(
                ax2, cmap=mpl.cm.viridis, norm=norm, orientation="horizontal"
            )
        )

        if hasattr(self.opti_data, "name"):
            fig.suptitle(self.opti_data.name)

        plt.show()

    def plot_metamodell_nvars(self):
        """
        Generates a Corner Plot of the different axes of the metamodel
        """

        dim = self.opti_data.x.shape[1]

        fig, axes = plt.subplots(nrows=dim, ncols=dim)

        x_data = self.opti_data.x

        x_step = np.zeros([self.num_ele, x_data.shape[1]])

        for i in range(x_data.shape[1]):
            x_step[:, i] = np.linspace(
                self.opti_data.limits[i, 0],
                self.opti_data.limits[i, 1],
                self.num_ele,
            )

        # XY = np.dstack(np.meshgrid(x_step))
        # fun1Vars = self.opti_data.sm.predict_values(XY.astype(np.float))

        images = []
        for i in range(x_data.shape[1]):
            for j in range(i + 1, x_data.shape[1]):
                images.append(
                    axes[j, i].scatter(
                        x_data[:, i], x_data[:, j], c="b", marker="o"
                    )
                )

        if hasattr(self.opti_data, "name"):
            fig.suptitle(self.opti_data.name)

        plt.show()

    def plot(self):
        """
        Calls a individual plot function depending on the provided data. If
        it is possible to plot the data in one or two dimensions, it will be
        plotted. If the visualization keyword is got set to "benchmarking"
        the metamodel will be visualized alongside the original model. A
        TreeOpt metamodel object (opt) can be visualized via the following
        statement:  from treeopt.treeOpt import visualize; vis =
        visualize.Visualize(opt); vis.plot()
        """

        if (
            self.opti_data.vis_keyword == "benchmarking"
            or self.opti_data.vis_keyword == "Benchmarking"
        ):
            if self.opti_data.x.shape[1] == 1:
                raise NotImplementedError(
                    "This function is not implemented at the moment"
                )
            elif self.opti_data.x.shape[1] == 2:
                self.plot_benchmark_2vars()
            else:
                raise NotImplementedError(
                    "This function is not implemented at the moment"
                )
        elif self.opti_data.vis_keyword == "extern":
            if self.opti_data.x.shape[1] == 1:
                self.plot_metamodell_1vars()
            elif self.opti_data.x.shape[1] == 2:
                self.plot_metamodell_2vars()
            else:
                self.plot_metamodell_nvars()
        else:
            raise KeyError("Wrong Keyword has been passed.")
