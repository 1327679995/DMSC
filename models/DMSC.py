import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers.DMSC_Layer import MultiScalePatchDecompositionBlock, TriadInteractionBlock, AdaptiveScaleRoutingMoEBlock
from layers.StandardNorm import Normalize


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.task_name = configs.task_name
        self.e_layers = configs.e_layers
        self.patch_len = configs.patch_len

        # For Model Analysis

        self.normalize_layer = Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)


        self.MPD = nn.ModuleList(
            [
                MultiScalePatchDecompositionBlock(
                    max_patch=self.patch_len, d_model=configs.d_model, dropout=configs.dropout
                )
                for i in range(configs.e_layers)
            ]
        )

        self.TIB = nn.ModuleList(
            [
                TriadInteractionBlock(
                    configs.enc_in, configs.dropout, configs.d_model, configs.d_ff,
                )
                for _ in range(configs.e_layers)
            ]
        )

        self.gate_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, 1),
                nn.Sigmoid()
            )
            for _ in range(configs.e_layers - 1)
        ])

        self.TIBres_projection = nn.ModuleList(
            [
                nn.Linear(configs.d_model, self.seq_len)
                for _ in range(configs.e_layers - 1)
            ]
        )

        self.MSMoE = AdaptiveScaleRoutingMoEBlock(
            configs.num_shared, configs.num_experts, configs.d_model, configs.d_ff, self.pred_len, self.e_layers, configs.top_k, configs.dropout, configs.balance_coeff, configs.use_res
        )

        self.res = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.pred_len),
        )



    def forecast(self, x_enc):
        # Normalization
        x_enc = self.normalize_layer(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1)
        # [batch_size, num_features, seq_len]
        x_res = x_enc.clone()
        outputs = []
        prev_output = None
        base_patch_len = None

        for i in range(self.e_layers):
            if prev_output is None:
                x_in = x_enc
            else:
                gate = self.gate_net[i-1](prev_output)
                projected_prev_output = self.TIBres_projection[i-1](prev_output)
                x_in = x_enc + gate * projected_prev_output
            # [batch_size, num_features, seq_len]
            mpd_output, base_patch_len = self.MPD[i](x_in, base_patch_len=base_patch_len, layer_idx=i)
            # [batch_size, num_features, patch_nums, d_model]
            x_out = self.TIB[i](mpd_output)
            # [batch_size, num_features, d_model]



            outputs.append(x_out)
            prev_output = x_out

        output, balance_loss = self.MSMoE(outputs)
        # [batch_size, num_features, pred_len]

        x_res = self.res(x_res)
        x_res = x_res.permute(0, 2, 1)
        output = output.permute(0, 2, 1)
        output = output + x_res

        output = self.normalize_layer(output, 'denorm')

        # [batch_size, pred_len, num_features]
        return output, balance_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':

            dec_out, balance_loss = self.forecast(x_enc)
            return dec_out, balance_loss
        else:
            raise ValueError('Other tasks implemented yet')






