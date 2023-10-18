import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class GpLayer(nn.Module):
    def __init__(self, d_model, time_feature_len, gp_seq_len,  enc_seq_len):
        super(GpLayer, self).__init__()
        self.dim_val = d_model
        half = int(d_model/2)
        self.gp_seq_len = gp_seq_len
        self.enc_seq_len = enc_seq_len

        self.fuse_src = nn.Linear(
            in_features=enc_seq_len*half,
            out_features=time_feature_len
            )
        self.recover = nn.Linear(
            in_features=time_feature_len,
            out_features=half
        )
        self.gp_transform = nn.Linear(
            in_features=gp_seq_len,
            out_features=enc_seq_len
        )

    def forward(self, src, timeFeature):
        half = int(self.dim_val/2)
        tmp = src[:, :, half:].flatten(1)  # B*(T*1/2D)
        key = torch.softmax(self.fuse_src(tmp), 1)  # B*P
        if self.gp_seq_len != self.enc_seq_len:
            fuse = key * self.gp_transform(timeFeature.transpose(0, 1)).transpose(0, 1).unsqueeze(1)
        else:
            fuse = key * timeFeature.unsqueeze(1)  # 逐元素乘，哈马达积：B*P * T*1*P = B*T*P
        out = torch.concat((src[:, :, :half], self.recover(fuse).transpose(0, 1)), dim=-1)
        return out, key


class GpBuilder(nn.Module):
    def __init__(self, d_model, d_low):
        super(GpBuilder, self).__init__()
        self.low_layer = nn.Linear(
            in_features=d_model,
            out_features=d_low
        )

    def forward(self, enc_out, gmm, gp, stl):
        low_enc = self.low_layer(enc_out)  # B*T*D -> B*T*d
        ser_re = torch.mean(low_enc, dim=-1)  # B*T
        decomp_x = torch.FloatTensor([stl(x)[0] for x in ser_re])  # B*T
        gmm = gmm.partial_fit(decomp_x)
        c_res = gmm.predict(decomp_x)
        if not gp.s_init:
            gp.build_pool_seasonal(decomp_x, c_res)
        else:
            gp.update_pool_seasonal(decomp_x, c_res)


class GpEncoder(nn.Module):
    def __init__(self, attn_layers, gp_builders, gp_layers, conv_layers=None, norm_layer=None):
        super(GpEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.gp_builders = nn.ModuleList(gp_builders)
        self.gp_layers = nn.ModuleList(gp_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, x_gp, gmm, gp_update_flag, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, gp_builder, conv_layer in zip(self.attn_layers, self.gp_builders, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer, gp_builder, gp_layer in zip(self.attn_layers, self.gp_builders, self.gp_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                if gp_update_flag:
                    gp_builder(x, gmm, x_gp)
                if x_gp != None:
                    x, key = gp_layer(x, x_gp.seasonal_pool)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if x_gp != None:
            return x, key, attns
        else:
            return x, None, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


