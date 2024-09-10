import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


def down_linear(in_features):
    conv_op = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, in_features // 4),
        nn.Sigmoid(),
        nn.Linear(in_features // 4, in_features // 16),
        nn.Sigmoid(),
        nn.Linear(in_features // 16, in_features // 64),
        nn.Sigmoid(),
        #nn.Linear(in_features // 64, in_features // 256),
        #nn.Sigmoid(),
        nn.Linear(in_features // 64, in_features // 128),
        nn.Sigmoid(),
        #nn.Linear(in_features // 128, in_features // 512),
        #nn.Sigmoid(),

    )
    return conv_op

def up_linear(out_features):
    conv_op = nn.Sequential(
        #nn.Linear(in_features, out_features // 128),
        #nn.Sigmoid(),
        #nn.Linear(out_features // 128, out_features // 64),
        #nn.Sigmoid(),
        #nn.Linear(out_features // 512, out_features // 128),
        #nn.Sigmoid(),
        nn.Linear(out_features // 128, out_features // 64),
        nn.Sigmoid(),
        nn.Linear(out_features // 64, out_features // 16),
        nn.Sigmoid(),
        nn.Linear(out_features // 16, out_features // 4),
        nn.Sigmoid(),
        nn.Linear(out_features // 4, out_features),
        nn.Sigmoid(),
        nn.Unflatten(1, (1, 1024))
    )
    return conv_op

class ViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height = 4
        image_width = 101

        self.to_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = image_height, p2 = image_width),
            nn.LayerNorm(channels * image_height * image_width),
            nn.Linear(channels * image_height * image_width, dim),
            nn.LayerNorm(dim),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)


    def forward(self, img):

        x = self.to_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class CAE(nn.Module):
    def __init__(
            self,
            *,
            decoder_dim = 128,
            decoder_depth=8,
            decoder_heads=8,
            decoder_dim_head=64
    ):
        super().__init__()

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = ViT(
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )
        #num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = self.encoder.to_embedding[0]
        self.emb = nn.Sequential(*self.encoder.to_embedding[1:])

        pixel_values_per_patch = self.encoder.to_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(1024, decoder_dim) #if encoder_dim != decoder_dim else nn.Identity()

        #print(f'encoder dim: {encoder_dim}, decoder dim: {decoder_dim}')
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head,
                                   mlp_dim=decoder_dim * 4)
        #self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.lin_transformation = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.down_flatten = down_linear(1024)
        self.up_flatten = up_linear(1024)


    def forward(self, img):
        # get patches

        patches = self.to_patch(img)
        #print(f'patch: {patches.shape}')
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.emb(patches)
        #print(f'emb_ {tokens.shape}')

        encoded_tokens = self.encoder.transformer(tokens)
        #print(f'encoded_tokens {encoded_tokens.shape}')

        down_features = self.down_flatten(encoded_tokens)
        up_features = self.up_flatten(down_features)
#
#
        ## project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
#
        decoder_tokens = self.enc_to_dec(up_features)
        #print(f'decoder_tokens {decoder_tokens.shape}')

#
        pred_pixel_values = self.lin_transformation(decoder_tokens)

#
        ## calculate reconstruction loss
#
        recon_loss = F.mse_loss(pred_pixel_values, patches)
        return recon_loss


mae = CAE(
    decoder_dim = 128,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

images = torch.randn(8, 1, 4, 101)

loss = mae.forward(images)
print(loss)