import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import VideoEncoder, AudioEncoder


class FusionModel(nn.Module):
    def __init__(self, d=256, d_proj=128, T=16, heads=1, r=5.0, dropout=0.1,
                 backbone='vit_base_patch16_224', pretrained=True, freeze_visual=False,
                 alpha_init=4.0, beta_init=-2.0, use_gate=True):
        super().__init__()
        self.T = T
        self.r = r
        self.use_gate = use_gate
        self.video = VideoEncoder(d, backbone, pretrained, freeze_visual)
        self.audio = AudioEncoder(d, t_out=T, dropout=dropout)
        self.proj_v = nn.Linear(d, d_proj, bias=False)
        self.proj_a = nn.Linear(d, d_proj, bias=False)
        self.attn_v = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.head = nn.Sequential(
            nn.Linear(2 * d + 1, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def forward(self, frames, mel):
        fv = self.video(frames)
        fa = self.audio(mel)

        pv = F.normalize(self.proj_v(fv), dim=-1)
        pa = F.normalize(self.proj_a(fa), dim=-1)
        s = (pv * pa).sum(-1)
        u = (1.0 - s) / 2.0

        fv_cross, _ = self.attn_v(fv, fa, fa)
        fa_cross, _ = self.attn_a(fa, fv, fv)
        plain = torch.cat([fv, fa], dim=-1)
        cross = torch.cat([fv_cross, fa_cross], dim=-1)

        if self.use_gate:
            g = torch.sigmoid(self.alpha * u + self.beta).unsqueeze(-1)
            z = (1.0 - g) * plain + g * cross
        else:
            g = torch.ones_like(u).unsqueeze(-1)
            z = cross

        u_bar = (torch.logsumexp(self.r * u, dim=1) - math.log(self.T)) / self.r
        h = torch.cat([z.mean(dim=1), u_bar.unsqueeze(-1)], dim=-1)
        logit = self.head(h).squeeze(-1)
        return {'logit': logit, 'u': u, 's': s, 'g': g.squeeze(-1), 'pv': pv, 'pa': pa}
