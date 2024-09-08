#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from tifffile import imread
import numpy as np
import pickle
import os
import pandas as pd
from scipy.optimize import curve_fit
from lpu3dnet.post_process.kr_process import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

root_dir = 'db'
vol_dim = 3

def calculate_error_metrics(predictions, true_value, metric='mse'):
    """
    Calculate error metrics for a list of predictions against a true value.

    Args:
    predictions (list): List of prediction values.
    true_value (float): The true value to compare against.
    metric (str): Specify 'mse' for Mean Squared Error or 'mae' for Mean Absolute Error.

    Returns:
    float: The calculated error metric.
    """
    # Repeat the true value to match the length of predictions
    true_values = [true_value] * len(predictions)
    
    if metric == 'mse':
        return mean_squared_error(true_values, predictions)
    elif metric == 'mae':
        return mean_absolute_error(true_values, predictions)
    else:
        raise ValueError("Metric must be 'mse' or 'mae'")

mse_swir_list_pred_all = []
mse_swir_list_compare_all = []

mse_sor_pred_all = []
mse_sor_list_compare_all = []


for ct_idx in range(1,6):

    with open(f'{root_dir}/sample_{ct_idx}/phys_results_{vol_dim}.pickle', 'rb') as file:
        # Deserialize the data from the file and assign it to a variable
        sim_results = pickle.load(file)


    for sample_idx in range(7):
        # df_all_compare = aggregate_kr(sim_results['compare'])
        num_pred = len(sim_results[sample_idx]['generate'])
        num_compare = len(sim_results['compare'])

        prediction = {'data':[],'par':[]}
        compare = {'data':[],'par':[]}


        for i in range(num_pred):
            if sim_results[sample_idx]['generate'][i]:
                df_pred = convert_dict_to_pd(sim_results[sample_idx]['generate'][i])
                corey_fit = Corey_fit(df_pred)
                kr_avg_pred,par,criteria = corey_fit.generate_kr_data()
                if criteria:
                    prediction['data'].append(kr_avg_pred)
                    prediction['par'].append(par)

        for i in range(num_compare):
            if not sim_results['compare'][i]:
                continue
            df_compare = convert_dict_to_pd(sim_results['compare'][i])
            corey_fit = Corey_fit(df_compare)
            kr_avg_compare,par,criteria = corey_fit.generate_kr_data()
            if criteria:
                compare['data'].append(kr_avg_compare)
                compare['par'].append(par)

        df_real = sim_results[sample_idx]['original']
        if not df_real:
            continue
        # fit the real data
        df_real = convert_dict_to_pd(df_real)
        corey_fit = Corey_fit(df_real)
        kr_real,par_real,criteria = corey_fit.generate_kr_data()
        if not criteria:
            continue

        prediction_data = prediction['data']
        compare_data = compare['data']

        num_samples = min(len(prediction_data), len(compare_data))

        irr_sw_pred = [x[0][1] for x in prediction['par']][:num_samples]
        irr_sw_compare = [x[0][1] for x in compare['par']][:num_samples]
        res_gas_pred = [x[1][1] for x in prediction['par']][:num_samples]
        res_gas_compare = [x[1][1] for x in compare['par']][:num_samples]


        
        # calculate error
        err_mse_swir_pred = calculate_error_metrics(
            irr_sw_pred,
            par_real[0][1], metric='mae'
            )
        err_mse_swir_compare = calculate_error_metrics(
            irr_sw_compare,
            par_real[0][1], metric='mae'
            )

        mse_swir_list_pred_all.append(err_mse_swir_pred)
        mse_swir_list_compare_all.append(err_mse_swir_compare)

# %%
plt.hist(mse_swir_list_pred_all, bins=20, alpha=0.5, label='Prediction')
plt.hist(mse_swir_list_compare_all, bins=20, alpha=0.5, label='Comparison')
plt.legend()
# %%
sum(mse_swir_list_pred_all)/len(mse_swir_list_pred_all), sum(mse_swir_list_compare_all)/len(mse_swir_list_compare_all)

# %%
