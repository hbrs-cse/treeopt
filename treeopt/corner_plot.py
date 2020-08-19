import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import numpy as np


class Visualize:
    def __init__(self, optimizer):
        """
        Initializes a visiualize Class
        :param optimizer: Object of the TreeOpt Class containing all
        information of the calculated metamodell
        :type optimizer: TreeOpt Class Object
        :return: Nothing
        :rtype: None

        """

        self.opti_data = optimizer
        self.num_ele = 100

    def set_n(self, n):
        self.num_ele = n

    def plot_benchmark_2vars(self):
        """
        Plots a benchmarking function on the right and the calculated
        metamodell on the left Outputs a matplotlib graph
        :return: Nothing
        :rtype: None

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

        fun1Vars = self.opti_data.simulate_problem(
            XY.astype(np.float)
        ).reshape(self.num_ele, self.num_ele)
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
        Plots a metamodell with two one input variable and one output variable
        :return: Nothing
        :rtype: None

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
        Plots a Metamodell with two input variables and one output variable
        :return: Nothing
        :rtype: None

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
        dim = self.opti_data.x.shape[1]
        
        fig, axes = plt.subplots(nrows=dim, ncols=dim)
        
        x_data = self.opti_data.x
        
        images = []
        
        for i in range(x_data.shape[1]):
            for j in range(i+1, x_data.shape[1]):
                images.append(axes[j,i].scatter(x_data[:,i], x_data[:,j], c="b", marker = "o"))
                
        if hasattr(self.opti_data, "name"):
            fig.suptitle(self.opti_data.name)
        
        plt.show()        
                
    def plot(self):
        """
        Calls a individual plot function depending on the provided data
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if (
            self.opti_data.sim_keyword == "benchmarking"
            or self.opti_data.sim_keyword == "Benchmarking"
        ):
            if self.opti_data.x.shape[1] == 1:
                print("Hier ist eine 2-D Benchmarking Funktion zu sehen")
            elif self.opti_data.x.shape[1] == 2:
                self.plot_benchmark_2vars()
            else:
                print("Hier ist eine N-Dim Benchmarking Funktion zu sehen")
        elif self.opti_data.sim_keyword == "extern":
            if self.opti_data.x.shape[1] == 1:
                self.plot_metamodell_1vars()
            elif self.opti_data.x.shape[1] == 2:
                self.plot_metamodell_2vars()
            else:
                print("Hier ist eine Höherdimensionale Funktion zu sehen")
        else:
            print("Fehler")