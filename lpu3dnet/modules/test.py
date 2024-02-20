#%%
def output_size(input_dim,padding,dilation,kernel_size,stride):
    return int(((input_dim + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1)

output_size(4,2,1,3,2)
# %%
