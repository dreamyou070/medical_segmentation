import torch.nn as nn
import torch
import einops


class SinglePositionalEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        x = x + pe
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x

class SinglePositionalEmbedding_concat(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)
        # [2] dimension reduction
        self.fc = nn.Linear(2*d_model, d_model)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        if x.dim() == 4:
            start_dim = 4
            x = einops.rearrange(x, 'b c h w -> b (h w) c')  # B,H*W,C
        b_size = x.shape[0]
        res = int(x.shape[1] ** 0.5)
        pe = self.positional_encodings.expand(b_size, -1, -1).to(x.device)
        # [1] concat query and position_embedder
        x = torch.cat([x, pe], dim=2)
        # [2] reshape query (dimension reduction)
        self.fc = self.fc.to(x.device)
        x = self.fc(x)
        if start_dim == 4:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=res, w=res)
        return x

class SinglePositionalRelativeEmbedding(nn.Module):

    def __init__(self,
                 max_len: int = 64 * 64,
                 d_model: int = 320, ):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.randn(1,max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):

        start_dim = 3
        # b,l,c -> b,d,l,c
        b, l, c = x.shape
        h = w = int(l ** 0.5)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        absolute_pe = self.positional_encodings.to(x.device)(x)
        absolute_pe = einops.rearrange(absolute_pe, 'b (h w) c -> b h w c', h=h, w=w)
        # absolute to relative
        relative_pe = torch.zeros_like(absolute_pe)

        for i in range(h):
            for j in range(w):
                # absolute_position = x[:,i,j,:]
                # get range
                re_pe = []
                for ii in range(i - 3, i + 3):
                    for jj in range(w - 3, j + 3):
                        if ii >= 0 and ii < h and jj >= 0 and jj < w:
                            new_value = absolute_pe[:, i, j, :] - absolute_pe[:, ii, jj, :]
                            re_pe.append(new_value)
                relative_pe[:, i, j, :] = torch.stack(re_pe).sum(dim=0)
        x = x + relative_pe
        return x
class AllPositionalEmbedding(nn.Module):

    layer_names_res_dim = {'down_blocks_0_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'down_blocks_0_attentions_1_transformer_blocks_0_attn2': (64, 320),

                           'down_blocks_1_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'down_blocks_1_attentions_1_transformer_blocks_0_attn2': (32, 640),

                           'down_blocks_2_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'down_blocks_2_attentions_1_transformer_blocks_0_attn2': (16, 1280),

                           'mid_block_attentions_0_transformer_blocks_0_attn2': (8, 1280),

                           'up_blocks_1_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_1_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_2_transformer_blocks_0_attn2': (16, 1280),

                           'up_blocks_2_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_1_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_2_transformer_blocks_0_attn2': (32, 640),

                           'up_blocks_3_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_1_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_2_transformer_blocks_0_attn2': (64, 320), }

    def __init__(self, pe_do_concat) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.positional_encodings = {}
        self.do_concat = pe_do_concat
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            if pe_do_concat :
                self.positional_encodings[layer_name] = SinglePositionalEmbedding_concat(max_len = res*res, d_model = dim)
            else :
                self.positional_encodings[layer_name] = SinglePositionalEmbedding(max_len = res*res, d_model = dim)

    def forward(self, x: torch.Tensor, layer_name):
        if layer_name in self.positional_encodings.keys() :
            position_embedder = self.positional_encodings[layer_name]
            output = position_embedder(x)
            return output
        else :
            return x

class AllPositionRelativeEmbedding(nn.Module):

    layer_names_res_dim = {'down_blocks_0_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'down_blocks_0_attentions_1_transformer_blocks_0_attn2': (64, 320),

                           'down_blocks_1_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'down_blocks_1_attentions_1_transformer_blocks_0_attn2': (32, 640),

                           'down_blocks_2_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'down_blocks_2_attentions_1_transformer_blocks_0_attn2': (16, 1280),

                           'mid_block_attentions_0_transformer_blocks_0_attn2': (8, 1280),

                           'up_blocks_1_attentions_0_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_1_transformer_blocks_0_attn2': (16, 1280),
                           'up_blocks_1_attentions_2_transformer_blocks_0_attn2': (16, 1280),

                           'up_blocks_2_attentions_0_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_1_transformer_blocks_0_attn2': (32, 640),
                           'up_blocks_2_attentions_2_transformer_blocks_0_attn2': (32, 640),

                           'up_blocks_3_attentions_0_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_1_transformer_blocks_0_attn2': (64, 320),
                           'up_blocks_3_attentions_2_transformer_blocks_0_attn2': (64, 320), }

    def __init__(self, pe_do_concat) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.positional_encodings = {}
        self.do_concat = pe_do_concat
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            #if pe_do_concat :
            #    self.positional_encodings[layer_name] = SinglePositionalRelativeEmbedding_concat(max_len = res*res, d_model = dim)
            #else :
            self.positional_encodings[layer_name] = SinglePositionalRelativeEmbedding(max_len = res*res, d_model = dim)

    def forward(self, x: torch.Tensor, layer_name):
        if layer_name in self.positional_encodings.keys() :
            position_embedder = self.positional_encodings[layer_name]
            output = position_embedder(x)
            return output
        else :
            return x