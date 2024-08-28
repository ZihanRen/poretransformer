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
volume_dim = 3
root_dir = 'db'

for ct_idx in range(6):

    
    img_data = load_data(ct_idx, volume_dim, root_dir)

    two_p_data = {}

    for sample_idx in range(8):

        two_p_data[sample_idx] = {}

        num_pred = len(img_data[sample_idx]['generate'])
        num_compare = len(img_data['compare'])

        pred = []
        compare = []

        for i in range(num_pred):
            img_tmp = img_data[sample_idx]['generate'][i]
            out = ps.metrics.two_point_correlation(img_tmp)
            pred.append(out)


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


# %% plot real 2p
# f = plt.figure(figsize=(10,10))
# plt.plot(two_p.distance,two_p.probability,label='real',linewidth=5)
# for i in range(num_pred):
#     plt.plot(pred[i].distance,pred[i].probability,c='r',alpha=0.3)
# for i in range(num_compare):
#     plt.plot(compare[i].distance,compare[i].probability,c='g',alpha=0.3)



# # %%
# import porespy as ps
# from matplotlib import pyplot as plt
# import numpy as np
# import pickle
# import os
# from edt import edt


# # read pickle file fiven ct and vol_dim
# def load_data(ct_idx, vol_dim, root_dir):
#     file_path = f'{root_dir}/sample_{ct_idx}/img_gen_vol_{vol_dim}.pkl'
#     with open(file_path, 'rb') as file:
#         img_results = pickle.load(file)
#     return img_results


# # gloabl variables
# volume_dim = 3
# root_dir = 'db'
# sample_idx = 0

# img_data = load_data(1, volume_dim, root_dir)
# img_tmp = img_data[sample_idx]['generate'][2]



# # %%
# f = plt.figure(figsize=(10,10))
# plt.imshow(img_tmp[:,:,0])

# bd = np.zeros_like(img_tmp)
# # construct the inlet
# bd[:,:,0] = 1

# bd *= img_tmp

# im_tmp = ps.filters.trim_disconnected_blobs(im=img_tmp, inlets=bd)
# dt = edt(img_tmp)
# f = plt.figure(figsize=(10,10))
# plt.imshow(img_tmp[:,:,0])
# # %%
# sizes = np.arange(int(dt.max())+1, 0, -1)
# mio = ps.filters.porosimetry(im=img_tmp, inlets=bd, sizes=sizes, mode='mio')

# # %%
# e = ps.metrics.pc_curve(im=img_tmp, sizes=mio, voxel_size=1e-6)
# sw = [1-x for x in e.snwp]
# fig, ax = plt.subplots()
# ax.step(
#     sw,np.array(np.log10(e.pc)), 'r--', where='post', markersize=20, linewidth=3, alpha=0.6, label='MIO'
#     )
# plt.xlabel('log(Capillary Pressure [Pa])')
# plt.ylabel('Non-wetting Phase Saturation')
# plt.legend()
# ax.xaxis.grid(True, which='both')
# %%
