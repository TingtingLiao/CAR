import torch
import torch.nn as nn
from vit_pytorch.vit import Transformer, repeat


class ViT(nn.Module):
    def __init__(self, *, num_views, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0.):
        # super().__init__()
        super(ViT, self).__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.num_views = num_views
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        _, C, N = img.shape
        x = img.contiguous().view(-1, self.num_views, C, N)
        x = x.permute(0, 3, 1, 2).contiguous().view(-1, self.num_views, C)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        x = x.view(-1, N, x.shape[1]).permute(0, 2, 1).contiguous()
        return x


if __name__ == '__main__':
    v = ViT(
        num_views=1,
        num_classes=2,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 4, 1024)

    preds = v(img)  # (1, 1000)
    print(preds.shape)

