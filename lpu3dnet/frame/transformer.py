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

    def loss_func(self, logits, target):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),ignore_index=-1)

    @torch.no_grad()
    def generate(self, idx, cond, max_new_tokens, temperature=1.0, top_k=None):
        self.model.eval()
        generate_idx = self.model.generate(idx,cond, max_new_tokens, temperature, top_k)

        return generate_idx

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
    

# test the module
if __name__ == "__main__":
    experiment_idx = 7
    @hydra.main(
    config_path=f"/journel/s0/zur74/LatentPoreUpscale3DNet/lpu3dnet/config/ex{experiment_idx}",
    config_name="transformer",
    version_base='1.2')

    def main(cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(OmegaConf.to_yaml(cfg))
        transformer_obj = Transformer(cfg).to(device)
        b = 10
        seq_len = int(27*8)
        cond_dim = 4
        patch_num = 8

        idx = torch.randint(0, 3000, (10, seq_len)).to(device)
        cond_info = torch.rand(b,patch_num,cond_dim).to(device).float()
        opt = transformer_obj.configure_optimizers(device)
        logits = transformer_obj(idx,cond_info)
        loss = transformer_obj.loss_func(logits, idx)
        print(idx.shape)
        print(cond_info.shape)

        # generation phase
        cond_info = torch.rand(b,patch_num,cond_dim).to(device).float()
        idx = torch.randint(0, 3000, (10, 18)).to(device)
        new_tokens = transformer_obj.generate(
            idx,
            cond_info,
            max_new_tokens=30,
            temperature=1.0,
            top_k=None
            )

    
    main()