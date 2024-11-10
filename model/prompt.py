import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, prompt_init='uniform', prompt_pool=True, pool_size=10, top_k=3, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()
        # each prompt: (key, value)
        self.length = length # length of each prompt
        self.embed_dim = embed_dim # prompt dimension
        self.prompt_pool = prompt_pool
        self.prompt_init = prompt_init
        self.pool_size = pool_size  # size of the prompt pool
        self.top_k = top_k  # select topK prompt value
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))  # (10, 5, 768)
                nn.init.uniform_(self.prompt, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def corr_loss(self, f_a, f_b):
        # empirical cross-correlation matrix
        f_a = f_a.reshape(-1, f_a.shape[-1])
        f_b = f_b.reshape(-1, f_b.shape[-1])
        f_a_norm = (f_a - f_a.mean(0)) / (f_a.std(0)+1e-6)
        f_b_norm = (f_b - f_b.mean(0)) / (f_b.std(0)+1e-6)
        c = torch.mm(f_a_norm.T, f_b_norm) / f_a_norm.size(0)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = self.off_diagonal(c).pow_(2).mean()
        loss = on_diag + 0.005 * off_diag

        return loss
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            x_embed_mean = x_embed
            prompt_key = self.prompt.mean(dim=1)
            prompt_norm = self.l2_normalize(prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['reduce_sim'] = reduce_sim  # similarity loss
            out['corr_loss'] = self.corr_loss(batched_prompt, batched_prompt)
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = batched_prompt  # B, top-K * length, dim

        return out
