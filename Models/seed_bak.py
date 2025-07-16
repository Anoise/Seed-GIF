import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from .model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
# Models.interpretable_diffusion.
from .s4 import S4

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, n_feat, pred_len, out_feat, act=nn.ReLU()):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=n_feat, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(out_feat, pred_len, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, n_feat + 1, 1) / (n_feat + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class S4Layer(nn.Module):
    def __init__(self, d_model, dropout=0.0, cross = False):
        super().__init__()
        self.layer = S4(
            d_model=d_model,
            d_state=128,
            bidirectional=True,
            dropout=dropout,
            transposed=True,
            postact=None,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = (
            nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()
        )
        if cross:
            self.trans = nn.Conv1d(d_model, d_model, kernel_size=1)
 
    def forward(self, x, x_enc=None):
        """
        Input x is shape (B, d_input, L)
        """
        if x_enc is not None:
            x = x + self.trans(x_enc)
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer: we ignore the state input and output for training
        z, _ = self.layer(z)
        # Dropout on the output of the layer
        z = self.dropout(z)
        # Residual connection
        x = z + x
        return x, None
    


    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x
        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)
        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)
        # Residual connection
        x = z + x
        return x, state


class S4Encoder(nn.Module):
    def __init__(self, n_embd, n_feat, dropout=0.1):
        super().__init__()
        self.s4layer = S4Layer(n_feat, dropout=dropout)

        self.time_ln =AdaLayerNorm(n_embd)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_linear1 = nn.Conv1d(
            in_channels=n_feat, out_channels=n_feat, kernel_size=1
        )
        self.out_linear2 = nn.Conv1d(
            in_channels=n_feat, out_channels=n_feat, kernel_size=1
        )
        self.feature_encoder = nn.Conv1d(n_embd, n_feat, kernel_size=1)

    def forward(self, x, t, mask =None, label_emb=None):
        x = self.time_ln(x, t)
        out, _ = self.s4layer(x)
        if label_emb is not None:
            out = out + self.feature_encoder(label_emb)
        out = self.tanh(out) * self.sigm(out)
        out1 = self.out_linear1(out)
        out2 = self.out_linear2(out)
        return out1 + x, out2


class S4Decoder(nn.Module):
    def __init__(self, n_embd, n_feat, n_node, dropout=0.1):
        super().__init__()
        self.s4layer = S4Layer(n_feat, dropout=dropout, cross=True)

        self.time_ln =AdaLayerNorm(n_embd)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        
        self.feature_encoder = nn.Conv1d(n_feat, n_embd, kernel_size=1)
        self.proj = nn.Conv1d(n_feat, n_feat * 2, 1)
        self.trend = TrendBlock(n_feat, n_node, n_embd)
        self.seasonal = FourierLayer(d_model=n_feat)
        
        self.ln = nn.LayerNorm(n_embd)
        self.trans = nn.Conv1d(n_feat, n_feat, kernel_size=1)
        self.linear = nn.Linear(n_embd, n_node)

    def forward(self, x, t, x_enc=None, mask =None, label_emb=None):
        x = self.time_ln(x, t)
        out, _ = self.s4layer(x, x_enc)
        if label_emb is not None:
            out = out + self.feature_encoder(label_emb)
        out = self.tanh(out) * self.sigm(out)
        
        out1, out2 = self.proj(out).chunk(2, dim=1)
        trend = self.trend(out1)
        season = self.seasonal(out2)
        out = out + self.trans(self.ln(out))
        m = torch.mean(x, dim=1, keepdim=True)
        
        return x - m, self.linear(m), trend, season
        

class Encoder(nn.Module):
    def __init__(self, n_layer=4, n_embd=1024, n_feat=128, dropout=0.1):
        super().__init__()
        print(n_embd,'n_embd')
        self.blocks = nn.Sequential(*[S4Encoder(n_embd=n_embd, n_feat=n_feat, dropout=dropout) for _ in range(n_layer)])

    def forward(self, x, t, padding_masks=None, label_emb=None):
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_feat, n_node, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            S4Decoder(n_embd=n_embd, n_feat=n_feat, n_node=n_node, dropout=dropout) 
            for _ in range(n_layer)])
        self.n_node = n_node
      
    def forward(self, x, t, x_enc, padding_masks=None, label_emb=None):
        b, c, e = x.shape
        mean = []
        season = torch.zeros((b, c, e), device=x.device)
        trend = torch.zeros((b, c, self.n_node), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, t, x_enc=x_enc, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season


class Seed(nn.Module):
    def __init__(
        self,
        n_node,
        seq_len,
        pred_len,
        n_layer_enc=2,
        n_layer_dec=2,
        n_embd=1024,
        dropout=0.1,
        conv_params=None,
    ):
        super().__init__()
        print('==>>', n_node, seq_len, pred_len, n_layer_enc, n_layer_dec)
        
        self.emb = Conv_MLP(n_node, n_embd, resid_pdrop=dropout)
        self.inverse = Conv_MLP(n_embd, n_node, resid_pdrop=dropout)
        if conv_params is None or conv_params[0] is None:
            if seq_len < 32 and n_node < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.proj_ssn = nn.Conv1d(n_embd, n_node, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.proj_mean = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer=n_layer_enc,n_embd=n_embd, n_feat=seq_len, dropout=dropout)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=dropout, max_len=seq_len)

        self.decoder = Decoder(n_layer=n_layer_dec,n_embd=n_embd, n_feat=seq_len, n_node=n_node, dropout=dropout)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=dropout, max_len=seq_len)
        
        self.trend_out = nn.Conv1d(seq_len, pred_len, kernel_size=1)
        self.season_out = nn.Conv1d(seq_len, pred_len, kernel_size=1)

    def forward(self, input, t, padding_masks=None, return_res=False):
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
        inp_dec = self.pos_dec(emb)
        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        _season = self.proj_ssn(season.transpose(1,-1)).transpose(1,-1) + res - res_m
        _trend = self.proj_mean(mean) + res_m + trend

        _trend, _season = self.trend_out(_trend), self.season_out(_season)

        if return_res:
            return _trend, self.combine_s(_season.transpose(1, 2)).transpose(1, 2), res - res_m

        return _trend, _season


if __name__ == '__main__':
    x = torch.rand(32, 36, 100)
    t = torch.randint(0, 1000, (len(x),)).long()
    net = Seed(100, 36, 48)
    
    trend, season = net(x, t)
    
    out = trend + season
    
    print(out.shape)
    
    