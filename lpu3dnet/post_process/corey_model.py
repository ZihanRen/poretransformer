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

root_dir = 'db'
ct_idx = 1
vol_dim = 3
with open(f'{root_dir}/sample_{ct_idx}/phys_results_{vol_dim}.pickle', 'rb') as file:
    # Deserialize the data from the file and assign it to a variable
    sim_results = pickle.load(file)


def truncated_kr(df,corey_fit):
    swir = corey_fit.swirr
    sorg = corey_fit.sor

    # find largest krw and krnw
    krw_max = df['krw'].max()
    krnw_max = df['krnw'].max()
    # truncated df based on the krw and krnw
    df = df[ (df['krw']<krw_max) & (df['krnw']<krnw_max) ]
    # truncated kr based on the swir and sorg
    # make sure sw in the range of swir and 1-sorg
    # df = df[(df['sw']>=swir) & (df['sw']<=1-sorg)]
    return df



#%%
# generate predictions ensemble
sample_idx = 6
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
            prediction['data'].append((truncated_kr(kr_avg_pred,corey_fit)))
            prediction['par'].append(par)



for i in range(num_compare):
    df_compare = convert_dict_to_pd(sim_results['compare'][i])
    corey_fit = Corey_fit(df_compare)
    kr_avg_compare,par,criteria = corey_fit.generate_kr_data()
    if criteria:
        compare['data'].append(truncated_kr(kr_avg_compare,corey_fit))
        compare['par'].append(par)

df_real = sim_results[sample_idx]['original']
# fit the real data
df_real = convert_dict_to_pd(df_real)
corey_fit = Corey_fit(df_real)
kr_real,par_real,criteria = corey_fit.generate_kr_data()

if criteria:
    kr_real = truncated_kr(kr_real,corey_fit)
print(criteria)


#%%

# you need to balance the number of samples
prediction_data = prediction['data']
compare_data = compare['data']
num_samples = min(len(prediction_data), len(compare_data))
irr_sw_pred = [x[0][1] for x in prediction['par']][:num_samples]
irr_sw_compare = [x[0][1] for x in compare['par']][:num_samples]
res_gas_pred = [x[1][1] for x in prediction['par']][:num_samples]
res_gas_compare = [x[1][1] for x in compare['par']][:num_samples]


from sklearn.metrics import mean_squared_error, mean_absolute_error
# calculate error
err_mse_swir_pred = calculate_error_metrics(
    irr_sw_pred,
    par_real[0][1], metric='mse'
    )
err_mse_swir_compare = calculate_error_metrics(
    irr_sw_compare,
    par_real[0][1], metric='mse'
    )

print(err_mse_swir_pred, err_mse_swir_compare)



f = plt.figure(figsize=(6, 4))
for i in range(num_samples):
    
    plt.plot(
        prediction_data[i]['sw'],
        prediction_data[i]['krnw'],
        'r-',linewidth=2,alpha=0.4
        )
    
    plt.plot(
        compare_data[i]['sw'],
        compare_data[i]['krnw'], 'y*',linewidth=1,alpha=0.4
        )
    plt.title('Exponential Fit to Relative Permeability')
    plt.xlabel('Water Saturation (Sw)')
plt.plot(kr_real['sw'], kr_real['krnw'], color='blue',linewidth=5, label='Actual Data')
plt.plot(df_real['sw'], df_real['kr_air'], color='blue', label='Actual Data')
plt.grid('True')
plt.show()




f = plt.figure(figsize=(6, 4))
for i in range(num_samples):
    plt.plot(
        prediction_data[i]['sw'],
        prediction_data[i]['krw'], 'r-',linewidth=2,alpha=0.4
        )
    plt.plot(compare_data[i]['sw'],
             compare_data[i]['krw'],
             'y*',
             linewidth=1,alpha=0.4
             )
    plt.title('Exponential Fit to Relative Permeability')
    plt.xlabel('Water Saturation (Sw)')
plt.plot(kr_real['sw'], kr_real['krw'], color='blue',linewidth=5, label='Actual Data')
plt.plot(df_real['sw'], df_real['kr_water'], color='blue', label='Actual Data')
plt.grid('True')
plt.show()


f = plt.figure(figsize=(10, 6))

# Plot histograms for both sets of predictions
plt.hist(irr_sw_pred, bins=20, alpha=0.5, label='Predictions')
plt.hist(irr_sw_compare[:num_samples], bins=20, alpha=0.5, label='Compare')

# Plot a vertical line for the true value
plt.axvline(x=par_real[0][1], color='r', linestyle='dashed', linewidth=2, label='True Value')

# Adding labels and title
plt.xlabel('Prediction Values')
plt.ylabel('Frequency')
plt.title('Histogram of Predictions with True Value Line')
plt.legend()
# Show the plot
plt.show()



# %%
