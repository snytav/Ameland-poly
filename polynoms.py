# make sure to import all of our modules
# sklearn package
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# dataframes
import pandas as pd
# computation
import numpy as np
# visualization
import matplotlib.pyplot as plt

# https://enjoymachinelearning.com/blog/multivariate-polynomial-regression-python/
def table2dataframe(deps,years,dists):
    years = np.array(years)
    years = years.reshape(years.shape[0],1)
    years = np.repeat(years,deps.shape[1],axis=1)
    dists = dists.reshape(dists.shape[0], 1)
    dists = np.repeat(dists.T,deps.shape[1],axis=0)

    years = years.reshape(deps.shape[0] * deps.shape[1])
    dists = dists.reshape(deps.shape[0] * deps.shape[1])

    deps = deps.reshape(deps.shape[0]*deps.shape[1])
    # df =pd.DataFrame(deps,columns = ['depth'])
    # df['year'] = time_year
    # df['dist'] = dist_m

    qq = 0
    return years,dists,deps





def polynomial_approximation():
    qq = 0