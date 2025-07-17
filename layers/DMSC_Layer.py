import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class MultiScalePatchDecompositionBlock(nn.Module):
    def __init__(self, max_patch, d_model=256, dropout=0.1):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.min_patch = 4
        self.max_patch = max_patch

        self.init_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
            nn.Dropout(p=dropout)
        )

        self.min_patch_param = nn.Parameter(torch.tensor(4.0))
        self.max_patch_param = nn.Parameter(torch.tensor(float(max_patch)))

        self.embed = nn.Sequential(
            nn.AdaptiveAvgPool1d(8),
            nn.Linear(8, d_model),
        )

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, base_patch_len=None, layer_idx=0):
        # [batch_size, num_features, seq_len]
        if base_patch_len is None:
            # [batch_size, num_features, seq_len]
            x1 = x.mean(dim=1, keepdim=True)
            scale_factor = self.init_net(x1)
            # [batch_size, seq_len]
            batch_scale = scale_factor.mean()
            min_patch = torch.clamp(self.min_patch_param, 2, 32).item()
            max_patch = torch.clamp(self.max_patch_param, 32, self.max_patch).item()
            patch_len = int(min_patch + (max_patch - min_patch) * batch_scale)

            base_patch_len = round(max(min_patch, min(self.max_patch, patch_len)))

        patch_len = max(self.min_patch, base_patch_len // (2 ** layer_idx))

        stride = max(1, patch_len // 2)
        padding = (stride - (x.size(-1) % stride)) % stride
        if padding > 0:
            x = F.pad(x, (0, padding), mode='replicate')
        x = x.unfold(dimension=-1, size=int(patch_len), step=int(stride))
        B, C, N, P = x.shape
        x = x.reshape(B * C, N, P)
        x = self.embed(x)
        x = x.reshape(B, C, N, -1)
        x = self.dropout(x)

        return x, base_patch_len


class TriadInteractionBlock(nn.Module):
    def __init__(self, num_features, dropout, embed_dim=256, d_c=32):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.d_c = d_c

        # Fusion Conv(Intra_Patch + Inter_Patch)
        # Intra-Patch
        self.intra_patch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        # Inter-Patch
        self.inter_patch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Cross-Variable
        self.cross_var = nn.Sequential(
            nn.Linear(embed_dim, d_c),
            nn.GELU(),
            nn.Linear(d_c, num_features),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )

        self.res_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # [batch_size, num_features, patch_nums, d_model]
        B, C, N, D = x.shape
        intra_input = x.permute(0, 1, 3, 2).reshape(B * C, D, N)
        intra_feat = self.intra_patch(intra_input)
        # [batch_size * num_features, d_model, patch_nums]

        inter_feat = self.inter_patch(intra_feat).squeeze(-1)
        # [batch_size * num_features, d_model, 1] -> [batch_size * num_features, d_model]
        var_feat = inter_feat.view(B, C, D)
        global_avg = var_feat.mean(dim=1, keepdim=True)
        cross_var = self.cross_var(global_avg).permute(0, 2, 1)
        # [batch_size, 1, num_features] -> [batch_size, num_features, 1]

        cross_feat = var_feat * cross_var
        intra_repr = intra_feat.mean(dim=-1).view(B, C, D)
        inter_repr = var_feat
        cross_repr = cross_feat

        fusion_input = torch.cat([intra_repr, inter_repr, cross_repr], dim=-1)
        gates = self.fusion_gate(fusion_input)

        g1, g2, g3 = gates.unbind(dim=-1)
        fused_feat = g1.unsqueeze(-1) * intra_repr + g2.unsqueeze(-1) * inter_repr + g3.unsqueeze(-1) * cross_repr

        ori_agg = x.mean(dim=2)
        res = self.res_adapter(ori_agg)
        output = self.norm(fused_feat + res)
        output = self.dropout(output)

        return output



class HierarchicalExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, expert_type='global', dropout=0.2):
        super().__init__()
        self.expert_type = expert_type

        if expert_type == 'global':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )

        elif expert_type == 'local':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.net(x)

class DynamicRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_shared, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.top_k = top_k

        # output for global and local weights
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        weights = self.router(x)
        global_weights = weights[:, :self.num_shared]
        local_weights = weights[:, self.num_shared:]

        local_weights, local_indices = torch.topk(local_weights, k=self.top_k, dim=-1)

        return global_weights, local_weights, local_indices, weights

class TemporalAwareWeighting(nn.Module):
    def __init__(self, input_dim, num_scales, dropout=0.2):
        super().__init__()
        self.num_scales = num_scales

        # temporal feature extractor
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # memory of history weight
        self.weight_memory = nn.Parameter(torch.zeros(1, num_scales))

        self.weight_calculator = nn.Sequential(
            nn.Linear((input_dim * num_scales) + num_scales, num_scales),
            nn.Tanh(),
            nn.Softplus(),
        )

    def forward(self, x_list, prev_weights=None):
        B, C, D = x_list[0].shape
        # [batch_size, num_features, embed_dim]
        scale_reprs = []
        for x in x_list:
            # extract representative feature
            scale_repr = torch.mean(x, dim=1)
            # apply temporal encoder
            encoded_repr = self.temporal_encoder(scale_repr)
            # [batch_size, embed_dim]
            scale_reprs.append(encoded_repr)

        # calculate each scale's importance
        concat_repr = torch.cat(scale_reprs, dim=-1)
        # [batch_size, num_scales * embed_dim]

        # fuse memory of history weight
        if prev_weights is not None:
            if prev_weights.shape[0] != B:
                memory_input = self.weight_memory.expand(B, -1)
            else:
                memory_input = prev_weights
            # [batch_size, num_scales]
        else:
            memory_input = self.weight_memory.expand(B, -1)
            # [batch_size, num_scales]
        combined_input = torch.cat([concat_repr, memory_input], dim=-1)
        raw_weights = self.weight_calculator(combined_input)
        weights = F.softmax(raw_weights, dim=-1)
        # [batch_size, num_scales]
        return weights


class AdaptiveScaleRoutingMoEBlock(nn.Module):
    def __init__(self, num_shared=2, num_experts=8, input_dim=256, hidden_dim=512, output_dim=256, num_scales=3, top_k=2, dropout=0.1, balance_coeff=0.1, use_res=False):
        super().__init__()
        assert top_k <= num_experts - num_shared, "top_k should be less than or equal to num_experts - num_shared(num_locals)"
        self.num_shared = num_shared
        self.num_experts = num_experts
        self.pred_len = output_dim
        self.top_k = top_k
        self.balance_coeff = balance_coeff
        self.use_res = use_res

        self.global_experts = nn.ModuleList([
            HierarchicalExpert(input_dim, hidden_dim, self.pred_len, expert_type='global', dropout=dropout)
            for _ in range(num_shared)
        ])

        self.local_experts = nn.ModuleList([
            HierarchicalExpert(input_dim, hidden_dim, self.pred_len, expert_type='local', dropout=dropout)
            for _ in range(num_experts - num_shared)
        ])

        # routing experts' weights
        self.router = DynamicRouter(input_dim, num_experts, num_shared, top_k=top_k)

        # aggregation(temporal aware)
        self.weight_calculator = TemporalAwareWeighting(self.pred_len, num_scales, dropout=dropout)

        self.output_layer = nn.Sequential(
            nn.Linear(self.pred_len, self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, self.pred_len),
            nn.Dropout(dropout),
        )

        if self.use_res:
            self.res = nn.Sequential(
                nn.Linear(input_dim, self.pred_len),
                nn.GELU(),
                nn.Linear(self.pred_len, self.pred_len),
            )

        # history weights(temporal dependency)
        self.prev_weights = None

    def calculate_balance_loss(self, all_expert_weights):
        combined_weights = torch.cat(all_expert_weights, dim=0)
        # [batch_size * num_features * num_scales, num_experts]
        entropy = -torch.mean(torch.sum(combined_weights * torch.log(combined_weights + 1e-8), dim=-1))

        return self.balance_coeff * entropy

    def forward(self, x_list):
        B, C, E = x_list[0].shape
        # [batch_size, num_features, embed_dim]

        all_preds=[]
        all_expert_weights = []

        if self.use_res:
            res = torch.zeros(B, C, self.pred_len).to(x_list[0].device)
        for x in x_list:
            x_flat = x.view(B * C, E)
            # [batch_size * num_features, embed_dim]
            global_weights, local_weights, local_indices, weights = self.router(x_flat)
            # global_weights: [batch_size * num_features, num_shared]
            # local_weights: [batch_size * num_features, top_k]
            # local_indices: [batch_size * num_features, top_k]
            # weights: [batch_size * num_features, num_experts]

            all_expert_weights.append(weights)

            global_input = x_flat.unsqueeze(1).expand(-1, self.num_shared, -1)
            # [batch_size * num_features, num_shared, embed_dim]
            global_outputs = torch.stack([
                expert(global_input[:, i]) for i, expert in enumerate(self.global_experts)
            ], dim=1)

            global_pred = torch.sum(global_outputs * global_weights.unsqueeze(-1), dim=1)
            # [batch_size * num_features, pred_len]

            local_outputs = torch.stack([expert(x_flat) for expert in self.local_experts], dim=1)
            # [batch_size * num_features, num_experts - num_shared, pred_len]
            selected_outputs = torch.gather(
                local_outputs,
                dim=1,
                index=local_indices.unsqueeze(-1).expand(-1, -1, local_outputs.shape[-1])
            )

            local_pred = torch.sum(selected_outputs * local_weights.unsqueeze(-1), dim=1)
            # [batch_size * num_features, pred_len]
            scale_pred = global_pred + local_pred
            scale_pred = scale_pred.view(B, C, self.pred_len)
            # [batch_size, num_features, pred_len]
            all_preds.append(scale_pred)
            # all scales predictions
            if self.use_res:
                res += self.res(x)

        # calculate load balance loss
        balance_loss = self.calculate_balance_loss(all_expert_weights)

        # calculate multiscale weights
        weights = self.weight_calculator(all_preds, self.prev_weights)
        # [batch_size, num_scales]
        # update history weights
        self.prev_weights = weights.detach()

        # aggregate predictions
        weighted_sum = torch.zeros_like(all_preds[0])
        # [batch_size, num_features, pred_len]
        for i, pred in enumerate(all_preds):
            weighted_sum += weights[:, i].unsqueeze(-1).unsqueeze(-1) * pred

        output = self.output_layer(weighted_sum)
        # [batch_size, num_features, pred_len]
        if self.use_res:
            output = output + res

        return output, balance_loss


