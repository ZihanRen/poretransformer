#%%
from lpu3dnet.modules import nanogpt
import torch
from torch import nn
import hydra
import os
from torch.nn import functional as F
import inspect


class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.cfg_architecture = cfg.architecture
        self.cfg_train = cfg.train

        self.model = nanogpt.GPT(
            self.cfg_architecture
        )


    def forward(self, idx, cond, inference=False):
        logits = self.model(idx,cond, inference=inference)
        return logits

    def loss_func_all(self, logits, target):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-1
            )

    def loss_func_last(self, logits, target):

        logits_last_patch = logits[:, -self.cfg_architecture.features_num:, :]
        # Extract the last patch of tokens from target
        target_last_patch = target[:, -self.cfg_architecture.features_num:]

        loss = F.cross_entropy(
            logits_last_patch.reshape(-1, logits_last_patch.size(-1)),
            target_last_patch.reshape(-1),
            ignore_index=-1
        )

        return loss

    def configure_optimizers(self,device):
        weight_decay = self.cfg_train.weight_decay
        learning_rate = self.cfg_train.learning_rate 
        betas = self.cfg_train.betas
        device_type = device

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def save_checkpoint(self,path):
        torch.save(self.state_dict(), path)
    
    # generate method removed as it's redundant - inference uses model.sample directly

# test the module
if __name__ == "__main__":
    
    with hydra.initialize(config_path="../config"):
        cfg_vqgan = hydra.compose(config_name="vqgan")
        cfg_transformer = hydra.compose(config_name="transformer")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    transformer_obj = Transformer(cfg_transformer).to(device)
    b = 10
    seq_len = int(64*8)
    cond_dim = 1
    patch_num = 8
    features_num = 64

    idx = torch.randint(0, 3000, (10, seq_len)).to(device)
    cond_info = torch.rand(b, patch_num*features_num, cond_dim).to(device).float()
    opt = transformer_obj.configure_optimizers(device)
    logits = transformer_obj(idx, cond_info)

    loss_all = transformer_obj.loss_func_all(logits, idx)
    loss_last = transformer_obj.loss_func_last(logits, idx)


    # testing sampling directly using the model.sample method
    sos_token = cfg_transformer.train.sos_token
    sos_tokens = torch.ones(1, features_num) * sos_token
    sos_tokens = sos_tokens.long().to(device)
    
    # sample a single token
    cond_test = torch.rand(1, features_num, cond_dim).to(device).float()
    with torch.no_grad():
        token_next = transformer_obj.model.sample(
            sos_tokens,
            cond_test,
            temperature=1.0,
            top_k=4,
            features_num=features_num
        )
    
    print(f"Generated token shape: {token_next.shape}")



# %%
