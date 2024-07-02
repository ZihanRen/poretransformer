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
from lpu3dnet.post_process.kr_process import Exponential_fit, aggregate_kr, convert_dict_to_pd


# root_dir = 'data_ref_hard'
root_dir = 'db'

ct_idx = 0
vol_dim = 3
with open(f'{root_dir}/sample_{ct_idx}/phys_results_{vol_dim}.pickle', 'rb') as file:
    # Deserialize the data from the file and assign it to a variable
    sim_results = pickle.load(file)


sample_idx = 3
df_all_compare = aggregate_kr(sim_results['compare'])
df_all_pred = aggregate_kr(sim_results[sample_idx]['generate'][:])
df_real = sim_results[sample_idx]['original']


#%% exponential fit
exp_fit = Exponential_fit(df_all_compare)
kr_avg_compare = exp_fit.generate_kr_data()
exp_fit = Exponential_fit(df_all_pred)
kr_avg_pred = exp_fit.generate_kr_data()


# compare the fitted data with the real data
plt.figure(figsize=(6, 4))
plt.plot(kr_avg_compare['sw'], kr_avg_compare['krnw'], 'r-', label='compare')
plt.plot(kr_avg_pred['sw'], kr_avg_pred['krnw'], 'g-', label='prediction')
plt.scatter(df_real['sw'], df_real['kr_air'], color='blue', label='Actual Data')
plt.title('Exponential Fit to Relative Permeability')
plt.xlabel('Water Saturation (Sw)')
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(kr_avg_compare['sw'], kr_avg_compare['krw'], 'r-', label='compare')
plt.plot(kr_avg_pred['sw'], kr_avg_pred['krw'], 'g-', label='prediction')
plt.scatter(df_real['sw'], df_real['kr_water'], color='blue', label='Actual Data')
plt.title('Exponential Fit to Relative Permeability')
plt.xlabel('Water Saturation (Sw)')
plt.legend()
plt.show()




#%% single data points
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


# root_dir = 'data_ref_hard'
root_dir = 'db'

ct_idx = 2
vol_dim = 3
with open(f'{root_dir}/sample_{ct_idx}/phys_results_{vol_dim}.pickle', 'rb') as file:
    # Deserialize the data from the file and assign it to a variable
    sim_results = pickle.load(file)




sample_idx = 3
# df_all_compare = aggregate_kr(sim_results['compare'])
df_all_pred = convert_dict_to_pd(sim_results[sample_idx]['generate'][4])
exp_fit = Exponential_fit(df_all_pred)
kr_avg_pred = exp_fit.generate_kr_data()

f = plt.figure(figsize=(6, 4))
plt.plot(kr_avg_pred['sw'], kr_avg_pred['krw'], 'g-', label='prediction')
plt.scatter(df_all_pred['sw'], df_all_pred['kr_water'], color='blue', label='Actual Data')
plt.title('Exponential Fit to Relative Permeability')
plt.xlabel('Water Saturation (Sw)')
plt.legend()
plt.show()


# %%
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

def clean_df(df):

    krw_max = df['krw'].iloc[-1]
    if krw_max < 0.2:
        return False
    return True
    

root_dir = 'db'
ct_idx = 3
vol_dim = 3
with open(f'{root_dir}/sample_{ct_idx}/phys_results_{vol_dim}.pickle', 'rb') as file:
    # Deserialize the data from the file and assign it to a variable
    sim_results = pickle.load(file)


#%%
# generate predictions ensemble
sample_idx = 7
# df_all_compare = aggregate_kr(sim_results['compare'])
num_pred = len(sim_results[sample_idx]['generate'])
num_compare = len(sim_results['compare'])

prediction = [] 
compare = []
for i in range(num_pred):
    df_pred = convert_dict_to_pd(sim_results[sample_idx]['generate'][i])
    exp_fit = Exponential_fit(df_pred)
    kr_avg_pred = exp_fit.generate_kr_data()
    if clean_df(kr_avg_pred):
        prediction.append(kr_avg_pred)
    


for i in range(num_compare):
    df_compare = convert_dict_to_pd(sim_results['compare'][i])
    exp_fit = Exponential_fit(df_compare)
    kr_avg_compare = exp_fit.generate_kr_data()
    if clean_df(kr_avg_compare):
        compare.append(kr_avg_compare)

df_real = sim_results[sample_idx]['original']
# fit the real data
df_real = convert_dict_to_pd(df_real)
exp_fit = Exponential_fit(df_real)
kr_real = exp_fit.generate_kr_data()


#%%
num_samples = min(len(prediction), len(compare))
f = plt.figure(figsize=(6, 4))
for i in range(num_samples):
    plt.plot(prediction[i]['sw'], prediction[i]['krw'], 'r-',linewidth=2,alpha=0.4)
    plt.plot(compare[i]['sw'], compare[i]['krw'], 'y*',linewidth=1,alpha=0.4)
    plt.title('Exponential Fit to Relative Permeability')
    plt.xlabel('Water Saturation (Sw)')
plt.plot(kr_real['sw'], kr_real['krw'], color='blue',linewidth=5, label='Actual Data')
plt.plot(df_real['sw'], df_real['kr_water'], color='blue', label='Actual Data')
plt.show()
# f = plt.figure(figsize=(6, 4))
# plt.plot(kr_avg_pred['sw'], kr_avg_pred['krw'], 'g-', label='prediction')
# plt.scatter(df_all_pred['sw'], df_all_pred['kr_water'], color='blue', label='Actual Data')
# plt.title('Exponential Fit to Relative Permeability')
# plt.xlabel('Water Saturation (Sw)')
# plt.legend()
# plt.show()

# %%
