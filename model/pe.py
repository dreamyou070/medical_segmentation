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

    def __init__(self) -> None:
        super().__init__()

        self.layer_dict = self.layer_names_res_dim
        self.positional_encodings = {}
        for layer_name in self.layer_dict.keys() :
            res, dim = self.layer_dict[layer_name]
            self.positional_encodings[layer_name] = SinglePositionalEmbedding(max_len = res*res, d_model = dim)

    def forward(self, x: torch.Tensor, layer_name):
        if layer_name in self.positional_encodings.keys() :
            position_embedder = self.positional_encodings[layer_name]
            output = position_embedder(x)
            return output
        else :
            return x