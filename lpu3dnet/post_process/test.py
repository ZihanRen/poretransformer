#%%
import pickle
import matplotlib.pyplot as plt
ct_idx = 0
volume_dim = 6

root_dir = 'db'
with open(f'{root_dir}/sample_{ct_idx}/two_p_data_{volume_dim}.pickle', 'rb') as file:
    # Deserialize the data from the file and assign it to a variable
    twp_p_results = pickle.load(file)


print(twp_p_results[0])
# %%
