import torch
import torch.nn as nn

from dyneformer_layers.DynEformer_EncDec import Decoder, DecoderLayer, GpEncoder, EncoderLayer, GpLayer
from dyneformer_layers.SelfAttention_Family import FullAttention, AttentionLayer
from dyneformer_layers.Embed import DataEmbedding


class StaticContextEmbedding(nn.Module):
    def __init__(self, dim_val, dim_w, dim_static, dropout):
        super().__init__()
        self.dim_val = dim_val
        half = int(dim_val/2)
        self.fuse_src = nn.Linear(
            in_features=dim_val,
            out_features=dim_w
            )
        self.static_embedding = nn.Linear(dim_static, dim_w)
        self.norm = nn.LayerNorm(dim_val)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, static_context):
        q = self.fuse_src(src)
        v = self.static_embedding(static_context)
        key = torch.softmax(q, -1)
        fuse = key * v.unsqueeze(1)
        return self.norm(src + self.dropout(fuse))


class DynEformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(DynEformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # dyneformer
        self.static_layer = StaticContextEmbedding(configs.d_model, configs.d_model, configs.dim_static, configs.dropout)
        self.if_padding = configs.if_padding

        # Encoder
        self.gp_encoder = GpEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [GpLayer(configs.d_model, configs.gp_len, configs.gp_seq_len, configs.enc_len) for l in range(configs.e_layers)],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_static, x_gp,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if x_static != None:
            enc_out = self.static_layer(enc_out, x_static)

        enc_out, key, attns = self.gp_encoder(enc_out, x_gp, attn_mask=enc_self_mask)

        if x_gp != None:
            global_padding = torch.matmul(key, x_gp.transpose(0, 1))  # dim = [batch_size, enc_len]
            if self.if_padding:  # padding
                if global_padding.shape[-1] >= self.pred_len:
                    x_dec[:, -self.pred_len:, 0] = global_padding[:, -self.pred_len:]
                else:
                    # Calculate the number of times global_padding needs to be repeated to reach or exceed pred_len
                    repeat_factor = (self.pred_len // global_padding.shape[-1]) + 1

                    # Extend global_padding by repeating it along its second dimension
                    extended_global_padding = global_padding.repeat(1, repeat_factor)

                    # Merge the extended global_padding into x_dec
                    x_dec[:, -self.pred_len:, 0] = extended_global_padding[:, -self.pred_len:]
            else:
                x_dec[:, -self.pred_len:, 0] = 0

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if x_static != None:
            dec_out = self.static_layer(dec_out, x_static)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
