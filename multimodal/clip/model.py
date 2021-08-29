import torch
import numpy as np
from torch import nn, Tensor



class LayerNorm(nn.LayerNorm):
    """Subclass nn.LayerNorm to handle fp16"""
    def forward(self, x: Tensor) -> Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class GELU(nn.Module):
    """Quick GELU"""
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, c1, ch, c2=None):
        super().__init__()
        self.c_fc = nn.Linear(c1, ch)
        self.gelu = GELU()
        self.c_proj = nn.Linear(ch, c2 or c1)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model*4)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: Tensor) -> Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: Tensor) -> Tensor:
        x += self.attention(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x 


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
        for _ in range(layers)])

    def forward(self, x: Tensor) -> Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution//patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x += self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        return x



class CLIP(nn.Module):
    def __init__(self, image_resolution: int, patch_size: int):
        super().__init__()
        embed_dim = 512
        # vision parameters
        vision_layers = 12
        vision_width = 768
        vision_heads = 12

        # text parameters
        context_length = 77
        vocab_size = 49408
        transformer_width = 512
        transformer_heads = 8
        transformer_layers = 12
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.visual = VisionTransformer(image_resolution, patch_size, vision_width, vision_layers, vision_heads, embed_dim)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, self.build_attention_mask())

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)   # zero out the lower diagonal
        return mask

    def encode_image(self, image: Tensor) -> Tensor:
        return self.visual(image.type(self.dtype))

    def encode_text(self, text: Tensor) -> Tensor:
        x = self.token_embedding(text).type(self.dtype) # [B, n_ctx, d_model]
        x += self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # consine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def _convert_to_fp16(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        m.weight.data = m.weight.data.half()
        if m.bias is not None:
            m.bias.data = m.bias.data.half()

    if isinstance(m, nn.MultiheadAttention):
        for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
            tensor = getattr(m, attr)
            if tensor is not None:
                tensor.data = tensor.data.half()

    for name in ['text_projection', 'proj']:
        if hasattr(m, name):
            attr = getattr(m, name)
            if attr is not None:
                attr.data = attr.data.half()



if __name__ == '__main__':
    model = CLIP(224, 32)
    jit_model = torch.jit.load('C:\\Users\\sithu\\Documents\\Projects\\multimodal\\checkpoints\\ViT-B-32.pt', map_location='cpu').eval()
    pre_dict = jit_model.state_dict()
    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in pre_dict:
    #         print(key)
    # model.apply(convert_to_fp16)
    model.load_state_dict(pre_dict, strict=False)
    model.eval()
    image = torch.randn(2, 3, 224, 224)
    text = torch.randint(0, 49408, (2, 77)).long()
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_logits, text_logits = model(image, text)
    print(image_features.shape)
    print(text_features.shape)
    print(image_logits.shape, text_logits.shape)
    

