#%%
import porespy as ps
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os


# read pickle file fiven ct and vol_dim
def load_data(ct_idx, vol_dim, root_dir):
    file_path = f'{root_dir}/sample_{ct_idx}/img_gen_vol_{vol_dim}.pkl'
    with open(file_path, 'rb') as file:
        img_results = pickle.load(file)
    return img_results


# gloabl variables
volume_dim = 6
root_dir = 'db'

for ct_idx in range(6):

    
    img_data = load_data(ct_idx, volume_dim, root_dir)

    two_p_data = {}

    for sample_idx in range(4):

        two_p_data[sample_idx] = {}

        

        num_pred = len(img_data[sample_idx]['generate'])
        num_compare = len(img_data['compare'])

        pred = []
        compare = []

        for i in range(num_pred):
            img_tmp = img_data[sample_idx]['generate'][i]
            try:
                out = ps.metrics.two_point_correlation(img_tmp)
                pred.append(out)
            except:
                print('Error in sample:', sample_idx, 'pred:', i)
                continue


        for i in range(num_compare):
            img_tmp = img_data['compare'][i]
            out = ps.metrics.two_point_correlation(img_tmp)
            compare.append(out)

        img_real = img_data[sample_idx]['original']
        two_p = ps.metrics.two_point_correlation(img_real)

        two_p_data[sample_idx]['real'] = two_p
        two_p_data[sample_idx]['pred'] = pred
        two_p_data[sample_idx]['compare'] = compare
        
    # save the data as pickle in current ct idx
    with open(f'{root_dir}/sample_{ct_idx}/two_p_data_{volume_dim}.pickle', 'wb') as file:
        pickle.dump(two_p_data, file)