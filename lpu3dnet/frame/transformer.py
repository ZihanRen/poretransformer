from lpu3dnet.modules import nanogpt
import torch
from torch import nn
from torchinfo import summary
import hydra
from omegaconf import OmegaConf
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.cfg_architecture.block_size else idx[:, -self.cfg_architecture.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


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

    def save_checkpoint(self,epoch):
        save_model_path = os.path.join(     
            self.save_path,
            f'transformer_epoch_{epoch}.pth'
            )
        torch.save(self.state_dict(), save_model_path)
    

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
        print(logits.shape)
        print(loss)
        print(opt)

        # idx = transformer_obj.generate(x, 1000, 1.0, 10)
        # print(idx.shape)

        # print(transformer_obj.model.get_num_params())
    
    main()