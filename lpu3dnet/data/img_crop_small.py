#%% load the image
import os
import pickle
import numpy as np
from tifffile import imread
from tifffile import imwrite
from lpu3dnet import init_yaml

def save_pickle(PATH,data):
    with open(PATH,'wb') as f:
        pickle.dump(data ,f)

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5



def np_to_tif(img,f_name):
    '''
    convert numpy to tif
    '''

    img_save = (img * 255).astype('uint8')
    # Save the 3D array as a 3D tif
    imwrite(f_name, img_save)

def tif_to_np(f_name):
    '''
    convert tif to numpy
    '''
    img = imread(f_name)
    img = img.astype('float32')/255
    return img>0.5

def delete_files_in_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    else:
        print(f"'{directory_path}' is not a valid directory")


#%% load large image
import math
img_list = []
for i in range(0,6):
    PATH = os.path.join(
        init_yaml.PATH['img_path']['main_vol'],
        f'main_{i}.tif'
        )

    img_list.append(
        tif_to_np(PATH)
        )


#%% find the optimal image interval for sub-sampling

im_size = 636
crop_s = 64
img_interval = 30

def output_img_num(im_size,crop_s,img_interval):
    img_num_one_side = ( ((im_size - 1)  - crop_s) / img_interval )
    img_num_one_side = math.floor(img_num_one_side)
    return img_num_one_side,img_num_one_side**3

print( "Total number of subvolume per large volume is {}".format(
    output_img_num(im_size,crop_s,img_interval)[1]
    ) 
    )
img_one,_ = output_img_num(im_size,crop_s,img_interval)


# %% beginning cropping images: 128*128*128 - subregion
# save cropped images into dict
def crop_by_idx(img_list,crop_s,img_interval,ct_idx):
    
    save_PATH = os.path.join(
        init_yaml.PATH['img_path']['sub_vol'],
        'ct_{}'.format(ct_idx)
        )
    
    # empty specific CT idx folder
    delete_files_in_directory(save_PATH)
    # create folder path give ct index
    os.makedirs(save_PATH,exist_ok=True)

    img = img_list[ct_idx]
    idx = 0

    for i in range(img_one):
        for j in range(img_one):
            for k in range(img_one):
                index_ib = img_interval*i
                index_ie = img_interval*i + crop_s
                index_jb = img_interval*j
                index_je = img_interval*j + crop_s
                index_kb = img_interval*k
                index_ke = img_interval*k + crop_s
                
                img_sample = img[
                index_ib:index_ie,
                index_jb:index_je,
                index_kb:index_ke
                ]

                img_path = os.path.join(save_PATH,f'{idx}.tif')
                np_to_tif(img_sample,img_path)

                idx += 1

im_size = 636
crop_s = 64
img_interval = 30

# begin cropping image
for i in range(0,6):
    crop_by_idx(img_list,crop_s,img_interval,i)


# %% test generated image
# import matplotlib.pyplot as plt
# test = img_samples['200']
# print(test.shape)
# print( 'The porosity of test image is {}'.format( ps.metrics.porosity(test) ) )
# plt.imshow(test[:,:,100])


#%% filter the library
# def filter(img,phi):
#     img_surface = []
#     filt_matrix = []
#     for index in [0,-1]:
#         img_sx = img[index,:,:]
#         img_sy = img[:,index,:]
#         img_sz = img[:,:,index]

#         img_surface.append(img_sx)
#         img_surface.append(img_sy)
#         img_surface.append(img_sz)

#     for img_t in img_surface:
#         phi_temp = ps.metrics.porosity(img_t)
#         filt_matrix.append(phi_temp>phi)

#     if False in filt_matrix:
#         return False
#     else:
#         return True

# img_list = []
# filter_phi = 0.07

# index = 0
# for img in img_samples.values():

#     if filter(img,filter_phi):
#         f_name = f'{index}.npy'
#         save_PATH = name+'-sub/'+ f_name
#         np.save(save_PATH,img)
#         index += 1


# %% test image load
# test = np.load(save_PATH)
# print(test.shape)
# print('Unique elements are {}'.format(np.unique(test)))


# resolution = 2.25e-6
# snow = ps.networks.snow(
# im=test,
# voxel_size=resolution)

# proj = op.io.PoreSpy.import_data(snow)

# pn,geo = proj[0],proj[1]

# %%
