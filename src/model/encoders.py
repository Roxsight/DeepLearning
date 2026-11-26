import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class VideoEncoder(nn.Module):
    def __init__(self, d=256, backbone='vit_base_patch16_224', pretrained=True, freeze=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.freeze = freeze
        self.proj = nn.Linear(self.backbone.num_features, d)
        if freeze:
            self.set_trainable(False)

    def set_trainable(self, flag):
        for p in self.backbone.parameters():
            p.requires_grad = flag

    def forward(self, frames):
        b = frames.shape[0]
        t = frames.shape[1]
        x = frames.flatten(0, 1)
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        out = self.proj(feat)
        return out.view(b, t, -1)


class AudioEncoder(nn.Module):
    def __init__(self, d=256, n_mels=128, n_steps=200, patch=16, stride=(16, 10),
                 layers=4, heads=4, t_out=16, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, d, kernel_size=patch, stride=stride)
        self.fp = (n_mels - patch) // stride[0] + 1
        self.tp = (n_steps - patch) // stride[1] + 1
        self.pos = nn.Parameter(torch.zeros(1, self.fp * self.tp, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.t_out = t_out

    def forward(self, mel):
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        mel = (mel - mean) / (std + 1e-5)
        x = self.patch_embed(mel.unsqueeze(1))
        b = x.shape[0]
        d = x.shape[1]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos
        x = self.norm(self.encoder(x))
        x = x.transpose(1, 2).reshape(b, d, self.fp, self.tp)
        x = x.mean(dim=2)
        x = F.adaptive_avg_pool1d(x, self.t_out)
        return x.transpose(1, 2)
