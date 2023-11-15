#%%
import time
start = time.time()

import numpy as np
codebook_db = np.load('latent_matrix.npy')
codebook_db = codebook_db.reshape(-1, 256)
# %% k means clustering of those vectors
import time
from sklearn.cluster import MiniBatchKMeans
import numpy as np

kmeans = MiniBatchKMeans(n_clusters=30000,
                          random_state=0,
                          batch_size=1024,
                          n_init="auto")
kmeans.fit(codebook_db)

end = time.time()
elipsed = (end - start) / (60*60)
print(f'Elapsed time: {elipsed} hours')

# %%
import pickle
with open('kmeans_30000.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

