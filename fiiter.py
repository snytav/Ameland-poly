# https://stats.stackexchange.com/questions/446398/multi-variable-nonlinear-scipy-curve-fit
import numpy, scipy, scipy.optimize
import matplotlib
from mpl_toolkits.mplot3d import  Axes3D
from matplotlib import cm # to colormap 3D surfaces from blue to red
import matplotlib.pyplot as plt



graphWidth = 800
graphHeight = 800
# 3D contour plot lines
numberOfContourLines = 16

def SurfacePlot(func, data, fittedParameters):

    f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=100)

    plt.grid(True)
    axes = Axes3D(f)

    # extract data from the single list
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter(x_data, y_data, z_data)  # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse)')  # add a title for surface plot
    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label
    axes.set_zlabel('Z Data')  # Z axis data label

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems


def ContourPlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=100)
    axes = f.add_subplot(111)

    # extract data from the single list
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = numpy.linspace(min(x_data), max(x_data), 20)
    yModel = numpy.linspace(min(y_data), max(y_data), 20)
    X, Y = numpy.meshgrid(xModel, yModel)

    Z = func(numpy.array([X, Y]), *fittedParameters)

    axes.plot(x_data, y_data, 'o')

    axes.set_title('Contour Plot')  # add a title for contour plot
    axes.set_xlabel('X Data')  # X axis data label
    axes.set_ylabel('Y Data')  # Y axis data label

    CS = matplotlib.pyplot.contour(X, Y, Z, numberOfContourLines, colors='k')
    matplotlib.pyplot.clabel(CS, inline=1, fontsize=10)  # labels for contours

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems


def ScatterPlot(data):
    f = plt.figure(figsize=(graphWidth / 100.0, graphHeight / 100.0), dpi=100)

    matplotlib.pyplot.grid(True)
    axes = Axes3D(f)

    # extract data from the single list
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    axes.scatter(x_data, y_data, z_data)

    axes.set_title('Scatter Plot (click-drag with mouse)')
    axes.set_xlabel('X Data')
    axes.set_ylabel('Y Data')
    axes.set_zlabel('Z Data')

    plt.show()
    plt.close('all')  # clean up after using pyplot or else there can be memory and process problems


def func(data, a1, a2, b):
    # extract data from the single list
    x1 = data[0]
    x2 = data[1]

    return (a1 / x1) + a2 * x2 + b


def fit2D(xData,yData,zData):
    from polynoms import table2dataframe
    #
    # xData = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    # yData = numpy.array([11.0, 12.1, 13.0, 14.1, 15.0, 16.1, 17.0, 18.1, 90.0])
    # zData = numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.0, 9.9])
#    =
    # place the data in a single list
    data = [xData, yData, zData]

#        if __name__ == "__main__":
    initialParameters = [1.0, 1.0, 1.0]  # these are the same as scipy default values in this example

    # here a non-linear surface fit is made with scipy's curve_fit()
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [xData, yData], zData, p0=initialParameters)

    ScatterPlot(data)
    SurfacePlot(func, data, fittedParameters)
    ContourPlot(func, data, fittedParameters)

    print('fitted parameters', fittedParameters)

    modelPredictions = func(data, *fittedParameters)

    absError = modelPredictions - zData

    SE = numpy.square(absError)  # squared errors
    MSE = numpy.mean(SE)  # mean squared errors
    RMSE = numpy.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (numpy.var(absError) / numpy.var(zData))
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)