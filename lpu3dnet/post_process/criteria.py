from cpgan.ooppnm import pnm_sim_old

def get_abs_k(img):
    data_pnm = pnm_sim_old.Pnm_sim(im=img)
    data_pnm.network_extract()
    if data_pnm.error == 1:
        raise ValueError('Error in network extraction')
    data_pnm.init_physics()
    data_pnm.get_absolute_perm()
    return data_pnm.data_tmp['kabs']



def filter_abs_k(img,threshold=90):
    kabs = get_abs_k(img)
    if kabs < threshold:
        return False
    else:
        return True