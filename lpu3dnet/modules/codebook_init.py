#%%
import torch
import pickle

class Codebook_init:
    def __init__(self, filepath: str):
        # Load the KMeans object from the specified file
        self.kmeans = self.load_kmeans(filepath)

        # Convert cluster centers to a PyTorch tensor
        self.codebook_emd = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)

    def load_kmeans(self, filepath: str):
        # Load and return the KMeans object from the given path using pickle
        try:
            with open(filepath, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            raise ValueError(f"Error loading KMeans object from {filepath}: {e}")
        



if __name__ == "__main__":
    path = '/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/finetune/kmeans_6000.pkl'
    codebook_init = Codebook_init(path)
    print(codebook_init.codebook_emd.shape)

# %%
