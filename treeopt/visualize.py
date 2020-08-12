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

        self.optiData = optimizer
        self.numEle = 100

    def setN(self, n):
        self.numEle = n

    def plotBenchmark2D(self):
        """
        Plots a benchmarking function on the right and the calculated
        metamodell on the left Outputs a matplotlib graph
        :return: Nothing
        :rtype: None

        """

        x_step = np.linspace(
            self.optiData.limits[0, 0], self.optiData.limits[0, 1], self.numEle
        )
        y_step = np.linspace(
            self.optiData.limits[1, 0], self.optiData.limits[1, 1], self.numEle
        )
        XY = np.dstack(np.meshgrid(x_step, y_step)).reshape(-1, 2)

        fun1Vars = self.optiData.benchmarkingProblem(
            XY.astype(np.float)
        ).reshape(self.numEle, self.numEle)
        fun2Vars = self.optiData.sm.predict_values(
            XY.astype(np.float)
        ).reshape(self.numEle, self.numEle)

        eqlines = 10
        cfsteps = 100
        v_min = np.floor(np.min([fun1Vars.reshape(-1)])).astype(np.float)
        v_max = np.ceil(np.max([fun1Vars.reshape(-1)])).astype(np.float)
        fig, axes = plt.subplots(nrows=1, ncols=2)

        images = []

        images.append(
            axes[0].contourf(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
                fun1Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes[0].contour(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
                fun1Vars,
                eqlines,
                colors="black",
            )
        )

        images.append(
            axes[1].contourf(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
                fun2Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes[1].contour(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
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
                    np.atleast_2d(self.optiData.x[: self.optiData.numDOE])[
                        :, 0
                    ].reshape(-1),
                    np.atleast_2d(self.optiData.x[: self.optiData.numDOE])[
                        :, 1
                    ].reshape(-1),
                    marker="x",
                    color="r",
                    label="Points-DOE",
                )
            )
            images.append(
                axes[i].scatter(
                    np.atleast_2d(self.optiData.x[self.optiData.numDOE :])[
                        :, 0
                    ].reshape(-1),
                    np.atleast_2d(self.optiData.x[self.optiData.numDOE :])[
                        :, 1
                    ].reshape(-1),
                    marker="x",
                    color="m",
                    label="Expanded Points",
                )
            )
            images.append(
                axes[i].scatter(
                    np.atleast_2d(self.optiData.current_best_point[0]),
                    self.optiData.current_best_point[1],
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

        if hasattr(self.optiData, "name"):
            fig.suptitle(self.optiData.name)

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

    def plotMetamodell(self):
        """
        Plots a Metamodell which with two input variables and one output
        variable
        :return: Nothing
        :rtype: None

        """

        x_step = np.linspace(
            self.optiData.limits[0, 0], self.optiData.limits[0, 1], self.numEle
        )
        y_step = np.linspace(
            self.optiData.limits[1, 0], self.optiData.limits[1, 1], self.numEle
        )
        XY = np.dstack(np.meshgrid(x_step, y_step)).reshape(-1, 2)

        fun1Vars = self.optiData.sm.predict_values(
            XY.astype(np.float)
        ).reshape(self.numEle, self.numEle)

        eqlines = 10
        cfsteps = 100
        v_min = np.floor(np.min([fun1Vars.reshape(-1)])).astype(np.float)
        v_max = np.ceil(np.max([fun1Vars.reshape(-1)])).astype(np.float)
        fig, axes = plt.subplots(nrows=1, ncols=1)

        images = []

        images.append(
            axes.contourf(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
                fun1Vars,
                cfsteps,
                alpha=1,
            )
        )
        images.append(
            axes.contour(
                XY[:, 0].reshape(self.numEle, self.numEle),
                XY[:, 1].reshape(self.numEle, self.numEle),
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
                np.atleast_2d(self.optiData.x[: self.optiData.numDOE])[
                    :, 0
                ].reshape(-1),
                np.atleast_2d(self.optiData.x[: self.optiData.numDOE])[
                    :, 1
                ].reshape(-1),
                marker="x",
                color="r",
            )
        )
        images.append(
            axes.scatter(
                np.atleast_2d(self.optiData.x[self.optiData.numDOE :])[
                    :, 0
                ].reshape(-1),
                np.atleast_2d(self.optiData.x[self.optiData.numDOE :])[
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

        if hasattr(self.optiData, "name"):
            fig.suptitle(self.optiData.name)

        plt.show()

    def plot(self):
        """
        Calls a individual plot function depending on the provided data
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if (
            self.optiData.simKeyword == "benchmarking"
            or self.optiData.simKeyword == "Benchmarking"
        ):
            self.plotBenchmark2D()
        elif self.optiData.simKeyword == "extern":
            self.plotMetamodell()
        else:
            print("Fehler")
