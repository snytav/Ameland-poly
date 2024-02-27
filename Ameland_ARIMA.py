import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
''

def ARIMA(df):
    model = pm.auto_arima(df.values, start_p=1, start_q=1,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=True,  # seaasonal
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    df.to_csv('ameland.csv')
    print(model.summary())
    auto_order = model.order

    model.plot_diagnostics(figsize=(21, 15))
    plt.show()
    qq = 0




def depth_ARIMA(depths_uniform,distances_uniform,num_years):
    dis_to_add = distances_uniform[0]
    df = pd.DataFrame(data=depths_uniform,
                      index=num_years,
                      columns=dis_to_add)

    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);

    ARIMA(df)
    qq = 0