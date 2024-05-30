"""
taken from Andrej Karpathy nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from lpu3dnet.modules.components import *

# for this model we use the same configuration as GPT-2
@dataclass
class GPTConfig:
    block_size: int = (2**3) * 64 # number of feature vectors per meta image * number of meta images
    vocab_size: int = 3000 # codebook size of vqvae
    n_layer: int = 8
    n_head: int = 12  # number of attention heads
    n_embd: int = 1080 # model dimension initial - 12*90. Each attention head has 90 dimensions
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    cond_dim: int = 1+3  # number of conditional features - phi + ijk
    cond_embd: int = 50  
    tokens_embd: int = 256  # embed tokens to certain dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.tokens_embd),
            wpe = nn.Embedding(config.block_size, config.tokens_embd),
            drop = nn.Dropout(config.dropout),
            cond_proj = nn.Linear(config.cond_dim, config.cond_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.attn_start = nn.Linear(config.cond_embd + config.tokens_embd, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, cond,inference=False):
        device = idx.device
        b, t = idx.shape  # batch size, total sequence length

        # Token and positional embeddings
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        all_emb = tok_emb + pos_emb

        # conditional vectors integration
        cond_rep = self.transformer.cond_proj(cond)

        # Concatenate everything and project to model dimension
        x = torch.cat((all_emb, cond_rep), dim=-1)
        x = self.attn_start(x)

        # Transformer processing
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # loss function should be claculated only based on next tokens
        if inference:
            return logits[:,-self.config.features_num:,:]
        return logits

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
    @torch.no_grad()
    def sample(self, token, cond, temperature=1.0, top_k=None,features_num=27):
        
        '''
        input: token and cond
        output: token with same length
        '''

        b,t,cond_dim = cond.shape
        self.eval()  # Ensure the model is in evaluation mode to disable dropout
        logits_gen = self.forward(token,cond,inference=True)
        logits_gen = logits_gen / temperature
        if top_k is not None:
            indices_to_remove = logits_gen < torch.topk(logits_gen, top_k, dim=-1).values.min(dim=-1, keepdim=True).values
            logits_gen[indices_to_remove] = -float('Inf')
        # Convert logits to probabilities
        probs = F.softmax(logits_gen, dim=-1)
        # (27,3001)
        probs = probs.view(-1, probs.size(-1))

        # Sample from the probability distributions for each of the predicted 27 token positions
        token_next_samples = torch.multinomial(probs, num_samples=1, replacement=True)
        # sampled indices (27,1)
        token_next = token_next_samples.view(-1, features_num)

        return token_next

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = GPTConfig()
    gpt = GPT(config).to(device)
    b = 10
    seq_len = int(27*8)
    cond_dim = 4
    patch_num = 8
    features_num = 27

    idx = torch.randint(0, 3000, (10, seq_len)).to(device)
    cond_info = torch.randn(b,patch_num*features_num,cond_dim).to(device).float()
    c = gpt(idx, cond_info,inference=False)
    print(c.shape)

    # autoregressive generation
    # idx = torch.randint(0, 3000, (10, 18)).to(device)
    # new_tokens = gpt.generate(
    #     idx,
    #     cond_info,
    #     max_new_tokens=30,
    #     temperature=1.0,
    #     top_k=None
    #     )
    # print(new_tokens.shape)
