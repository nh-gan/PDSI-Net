
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def MLP(dims, dropout=0.1, activation=nn.GELU):
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class ContinuousTimeEncoder(nn.Module):
    """Continuous sin-linear time encoding for irregular timestamps.

    t: [..., 1] -> [..., D]
    """

    def __init__(self, d_model: int):
        super().__init__()
        periodic_dim = d_model // 2
        self.w = nn.Parameter(torch.randn(1, periodic_dim) * 0.1)
        self.p = nn.Parameter(torch.zeros(1, periodic_dim))
        self.linear = nn.Linear(1, d_model - periodic_dim)

    def forward(self, t: Tensor) -> Tensor:
        t = torch.nan_to_num(t)
        periodic = torch.sin(t * self.w + self.p)
        return torch.cat([periodic, self.linear(t)], dim=-1)


class MaskedPointNetPooling(nn.Module):
    """PointNet symmetric pooling with missing-value mask.

    points: [B,P,M,D]
    valid:  [B,P,M]
    out:    [B,P,D]
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.fuse = MLP([2 * d_model, 2 * d_model, d_model], dropout)

    def forward(self, points: Tensor, valid: Tensor) -> Tensor:
        valid_f = valid.unsqueeze(-1).to(points.dtype)
        count = valid_f.sum(dim=2).clamp_min(1.0)
        mean_pool = (points * valid_f).sum(dim=2) / count

        very_neg = torch.finfo(points.dtype).min / 4
        masked_points = points.masked_fill(~valid.unsqueeze(-1), very_neg)
        max_pool = masked_points.max(dim=2).values
        has_point = valid.any(dim=2, keepdim=True)
        max_pool = torch.where(has_point, max_pool, torch.zeros_like(max_pool))
        return self.fuse(torch.cat([mean_pool, max_pool], dim=-1))


class FourierVariableMixer(nn.Module):
    """Low-frequency Fourier mixer over the variable axis.

    x: [B,N,D] -> [B,N,D]
    It keeps the PointNet/patch/discrete branch intact, then adds global
    variable-frequency interaction without building an O(N^2) graph.
    """

    def __init__(self, d_model: int, num_variables: int, dropout: float, modes: int = 16):
        super().__init__()
        self.modes = max(1, min(int(modes), num_variables // 2 + 1))
        self.weight_real = nn.Parameter(torch.randn(self.modes, d_model) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(self.modes, d_model) * 0.02)
        self.proj = MLP([d_model, d_model, d_model], dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,N,D]
        residual = x
        n_vars = x.shape[1]
        x_ft = torch.fft.rfft(x.float(), dim=1)  # [B,F,D], complex64/complex128
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x_ft.shape[1])
        weight = torch.complex(self.weight_real[:modes], self.weight_imag[:modes]).unsqueeze(0)
        out_ft[:, :modes, :] = x_ft[:, :modes, :] * weight
        mixed = torch.fft.irfft(out_ft, n=n_vars, dim=1).to(dtype=x.dtype)  # [B,N,D]
        mixed = self.proj(mixed)
        return self.norm(residual + mixed)


class PointNetPatchASTGIEncoder(nn.Module):
    """PointNet-style patch ASTGI encoder.

    Innovation:
    1. Split irregular history into time patches.
    2. Treat observed values in each patch as a discrete point set.
    3. Use point-wise MLP and symmetric pooling to obtain robust patch global features.
    4. Feed patch global features back to variable tokens, then model patch sequence.
    """

    def __init__(
        self,
        num_variables: int,
        pred_len: int,
        d_model: int,
        dropout: float,
        patch_len: int = 4,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.pred_len = pred_len
        self.d_model = d_model
        self.patch_len = max(2, int(patch_len))

        self.value_encoder = MLP([1, d_model, d_model], dropout)
        self.abs_time_encoder = ContinuousTimeEncoder(d_model)
        self.rel_time_encoder = ContinuousTimeEncoder(d_model)
        self.variable_embedding = nn.Embedding(num_variables, d_model)
        self.mask_embedding = nn.Embedding(2, d_model)

        self.point_mlp = MLP([4 * d_model, 2 * d_model, d_model, d_model], dropout)

        # Local ASTGI-style interaction inside each patch.  The previous
        # PointNet pooling was permutation-invariant but too weak: variables in
        # the same short patch could not exchange information before pooling.
        self.local_q = nn.Linear(d_model, d_model)
        self.local_k = nn.Linear(d_model, d_model)
        self.local_v = nn.Linear(d_model, d_model)
        self.local_update = MLP([2 * d_model, 2 * d_model, d_model], dropout)
        self.local_norm = nn.LayerNorm(d_model)
        self.global_pool = MaskedPointNetPooling(d_model, dropout)

        self.var_pool_fuse = MLP([3 * d_model, 2 * d_model, d_model], dropout)
        self.global_feedback = MLP([3 * d_model, 2 * d_model, d_model], dropout)
        self.patch_norm = nn.LayerNorm(d_model)

        self.temporal_gru = nn.GRU(d_model, d_model, batch_first=True)

        # Original discrete observation branch.  It keeps the unpatched ASTGI
        # observation dynamics so patch compression does not erase useful
        # abrupt clinical changes in P12.
        self.obs_token_mlp = MLP([4 * d_model, 2 * d_model, d_model], dropout)
        self.obs_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.obs_temporal_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.obs_norm = nn.LayerNorm(d_model)
        self.branch_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.branch_norm = nn.LayerNorm(d_model)
        self.temporal_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.temporal_norm = nn.LayerNorm(d_model)

        self.spatial_q = nn.Linear(d_model, d_model)
        self.spatial_k = nn.Linear(d_model, d_model)
        self.spatial_v = nn.Linear(d_model, d_model)
        self.spatial_update = MLP([2 * d_model, 2 * d_model, d_model], dropout)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.fourier_mixer = FourierVariableMixer(d_model, num_variables, dropout, modes=16)

        self.horizon_embedding = nn.Parameter(torch.randn(1, pred_len, 1, d_model) * 0.02)

        # Recent-slope branch: use the last two valid observations and their
        # real time gap to expose short-term trend. This targets large forecast
        # errors on P12 without replacing the patch/point ASTGI backbone.
        self.trend_encoder = MLP([2, d_model, d_model], dropout)
        self.trend_scale = nn.Parameter(torch.tensor(0.1))

        # The old hard residual head returned ``last_value + delta``. That is
        # stable, but it strongly anchors the model to the last-observation
        # baseline and makes large encoder changes look like tiny metric moves.
        # This dual head keeps the residual path while adding a direct forecast
        # path, and learns per-sample/per-horizon weights between them.
        self.residual_head = MLP([5 * d_model + 1, 2 * d_model, d_model, 1], dropout)
        self.direct_head = MLP([5 * d_model + 1, 2 * d_model, d_model, 1], dropout)
        self.output_gate = MLP([5 * d_model + 1, d_model, 1], dropout)
        nn.init.constant_(self.output_gate[-1].bias, 1.0)

    def _pad_to_patch(self, x: Tensor, mask: Tensor, time: Tensor):
        bsz, length, _ = x.shape
        pad_len = (self.patch_len - length % self.patch_len) % self.patch_len
        if pad_len == 0:
            return x, mask, time
        x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
        mask = F.pad(mask, (0, 0, 0, pad_len), value=False)
        pad_time = time[:, -1:, :].expand(bsz, pad_len, 1)
        return x, mask, torch.cat([time, pad_time], dim=1)

    def _last_observed_value(self, x: Tensor, mask: Tensor) -> Tensor:
        bsz, length, _ = x.shape
        order = torch.arange(length, device=x.device).view(1, length, 1)
        index = torch.where(mask, order, torch.full_like(order, -1))
        last_index = index.max(dim=1).values.clamp_min(0)
        gathered = x.gather(1, last_index.unsqueeze(1)).squeeze(1)
        return torch.where(mask.any(dim=1), gathered, torch.zeros_like(gathered))

    def _last_observed_trend(self, x: Tensor, time: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return recent physical-time slope and last observation time.

        x: [B,L,N], time: [B,L,1], mask: [B,L,N]
        slope: [B,N], last_time: [B,N]
        """
        bsz, length, num_variables = x.shape
        order = torch.arange(length, device=x.device).view(1, length, 1)
        valid_index = torch.where(mask, order, torch.full_like(order, -1))
        last_index = valid_index.max(dim=1).values.clamp_min(0)

        before_last = mask & (order < last_index.unsqueeze(1))
        prev_index = torch.where(before_last, order, torch.full_like(order, -1)).max(dim=1).values.clamp_min(0)

        last_value = x.gather(1, last_index.unsqueeze(1)).squeeze(1)
        prev_value = x.gather(1, prev_index.unsqueeze(1)).squeeze(1)
        time_by_var = time.expand(-1, -1, num_variables)
        last_time = time_by_var.gather(1, last_index.unsqueeze(1)).squeeze(1)
        prev_time = time_by_var.gather(1, prev_index.unsqueeze(1)).squeeze(1)

        has_obs = mask.any(dim=1)
        has_two = before_last.any(dim=1) & has_obs
        default_last_time = time[:, -1, :].expand(bsz, num_variables)
        last_time = torch.where(has_obs, last_time, default_last_time)

        delta_t = (last_time - prev_time).abs().clamp_min(1e-3)
        slope = (last_value - prev_value) / delta_t
        slope = torch.where(has_two, slope, torch.zeros_like(slope))
        return torch.nan_to_num(slope).clamp(-10.0, 10.0), torch.nan_to_num(last_time)

    def _pool_over_patch_time(self, point_state: Tensor, mask: Tensor) -> Tensor:
        # point_state: [B,P,S,N,D], mask: [B,P,S,N] -> [B,P,N,D]
        mask_f = mask.unsqueeze(-1).to(point_state.dtype)
        count = mask_f.sum(dim=2).clamp_min(1.0)
        mean_state = (point_state * mask_f).sum(dim=2) / count

        very_neg = torch.finfo(point_state.dtype).min / 4
        masked_state = point_state.masked_fill(~mask.unsqueeze(-1), very_neg)
        max_state = masked_state.max(dim=2).values
        has_value = mask.any(dim=2, keepdim=False).unsqueeze(-1)
        max_state = torch.where(has_value, max_state, torch.zeros_like(max_state))
        return self.var_pool_fuse(torch.cat([mean_state, max_state, mean_state - max_state], dim=-1))

    def _encode_observation_branch(self, x: Tensor, time: Tensor, valid: Tensor) -> Tensor:
        # x: [B,L,N], time: [B,L,1], valid: [B,L,N] -> [B,N,D]
        bsz, length, num_variables = x.shape
        var_ids = torch.arange(num_variables, device=x.device)
        value_e = self.value_encoder(x.unsqueeze(-1))
        time_e = self.abs_time_encoder(time).unsqueeze(2).expand(-1, -1, num_variables, -1)
        var_e = self.variable_embedding(var_ids).view(1, 1, num_variables, self.d_model).expand_as(value_e)
        mask_e = self.mask_embedding(valid.long())
        token = self.obs_token_mlp(torch.cat([value_e, time_e, var_e, mask_e], dim=-1))
        token = token * valid.unsqueeze(-1).to(token.dtype)

        obs_in = token.permute(0, 2, 1, 3).reshape(bsz * num_variables, length, self.d_model)
        obs_out, _ = self.obs_gru(obs_in)
        obs_out = obs_out.view(bsz, num_variables, length, self.d_model)
        token_bnld = token.permute(0, 2, 1, 3)

        order = torch.arange(length, device=x.device).view(1, 1, length)
        valid_bnl = valid.permute(0, 2, 1)
        last_idx = torch.where(valid_bnl, order, torch.full_like(order, -1)).max(dim=-1).values.clamp_min(0)
        gather_idx = last_idx.view(bsz, num_variables, 1, 1).expand(-1, -1, 1, self.d_model)
        obs_last = obs_out.gather(2, gather_idx).squeeze(2)
        token_last = token_bnld.gather(2, gather_idx).squeeze(2)
        has_obs = valid.any(dim=1).unsqueeze(-1)
        obs_last = torch.where(has_obs, obs_last, torch.zeros_like(obs_last))
        token_last = torch.where(has_obs, token_last, torch.zeros_like(token_last))
        gate = self.obs_temporal_gate(torch.cat([token_last, obs_last], dim=-1))
        return self.obs_norm(token_last + gate * obs_last)

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor, y_mark: Tensor) -> Tensor:
        # x: [B,L,N], x_mark: [B,L,Tc], x_mask: [B,L,N], y_mark: [B,Q,Tc]
        bsz, _, num_variables = x.shape
        safe_x = torch.nan_to_num(x)
        time = torch.nan_to_num(x_mark[..., 0:1])
        y_time = torch.nan_to_num(y_mark[..., 0:1])
        valid = x_mask.bool() & torch.isfinite(x)
        last_value = self._last_observed_value(safe_x, valid)  # [B,N]
        recent_slope, last_time = self._last_observed_trend(safe_x, time, valid)  # [B,N], [B,N]
        obs_node_state = self._encode_observation_branch(safe_x, time, valid)  # [B,N,D]

        safe_x, valid, time = self._pad_to_patch(safe_x, valid, time)
        _, padded_len, _ = safe_x.shape
        num_patches = padded_len // self.patch_len

        value = safe_x.view(bsz, num_patches, self.patch_len, num_variables)
        mask = valid.view(bsz, num_patches, self.patch_len, num_variables)
        abs_time = time.view(bsz, num_patches, self.patch_len, 1)
        patch_center = abs_time.mean(dim=2, keepdim=True)
        rel_time = abs_time - patch_center

        value_e = self.value_encoder(value.unsqueeze(-1))
        abs_time_e = self.abs_time_encoder(abs_time).unsqueeze(3).expand(-1, -1, -1, num_variables, -1)
        rel_time_e = self.rel_time_encoder(rel_time).unsqueeze(3).expand(-1, -1, -1, num_variables, -1)
        var_ids = torch.arange(num_variables, device=x.device)
        var_e = self.variable_embedding(var_ids).view(1, 1, 1, num_variables, self.d_model)
        mask_e = self.mask_embedding(mask.long())

        point_input = torch.cat([value_e, abs_time_e + rel_time_e, var_e.expand_as(value_e), mask_e], dim=-1)
        point_state = self.point_mlp(point_input)  # [B,P,S,N,D]
        point_state = point_state * mask.unsqueeze(-1).to(point_state.dtype)

        # Patch-local discrete interaction: [B*P*S,N,D].  This is the compact
        # ASTGI-like step inside each tPatch-style block.
        local_state = point_state.reshape(bsz * num_patches * self.patch_len, num_variables, self.d_model)
        local_mask = mask.reshape(bsz * num_patches * self.patch_len, num_variables)
        lq = self.local_q(local_state)
        lk = self.local_k(local_state)
        lv = self.local_v(local_state)
        local_score = torch.matmul(lq, lk.transpose(-1, -2)) / math.sqrt(self.d_model)
        local_score = local_score.masked_fill(~local_mask.unsqueeze(1), -1e4)
        local_attn = torch.softmax(local_score.float(), dim=-1).to(local_state.dtype)
        local_msg = torch.matmul(local_attn, lv)
        local_update = self.local_update(torch.cat([local_state, local_msg], dim=-1))
        local_state = self.local_norm(local_state + local_update)
        local_state = local_state * local_mask.unsqueeze(-1).to(local_state.dtype)
        point_state = local_state.view(bsz, num_patches, self.patch_len, num_variables, self.d_model)

        flat_points = point_state.reshape(bsz, num_patches, self.patch_len * num_variables, self.d_model)
        flat_valid = mask.reshape(bsz, num_patches, self.patch_len * num_variables)
        patch_global = self.global_pool(flat_points, flat_valid)  # [B,P,D]

        var_state = self._pool_over_patch_time(point_state, mask)  # [B,P,N,D]
        global_to_var = patch_global.unsqueeze(2).expand(-1, -1, num_variables, -1)
        var_static = self.variable_embedding(var_ids).view(1, 1, num_variables, self.d_model).expand(bsz, num_patches, -1, -1)
        patch_state = self.global_feedback(torch.cat([var_state, global_to_var, var_static], dim=-1))
        patch_state = self.patch_norm(var_state + patch_state)

        temporal_in = patch_state.permute(0, 2, 1, 3).reshape(bsz * num_variables, num_patches, self.d_model)
        temporal_out, _ = self.temporal_gru(temporal_in)
        last_temporal = temporal_out[:, -1]
        last_patch = temporal_in[:, -1]
        gate = self.temporal_gate(torch.cat([last_patch, last_temporal], dim=-1))
        node_state = self.temporal_norm(last_patch + gate * last_temporal).view(bsz, num_variables, self.d_model)

        q = self.spatial_q(node_state)
        k = self.spatial_k(node_state)
        v = self.spatial_v(node_state)
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        relation = torch.softmax(score.float(), dim=-1).to(node_state.dtype)
        msg = torch.matmul(relation, v)
        node_state = self.spatial_norm(node_state + self.spatial_update(torch.cat([node_state, msg], dim=-1)))
        node_state = self.fourier_mixer(node_state)

        # Late fusion of patch representation and original discrete observation
        # representation.  Patch branch captures block-level robust structure;
        # observation branch preserves point-level dynamics.
        branch_gate = self.branch_gate(torch.cat([node_state, obs_node_state], dim=-1))
        node_state = self.branch_norm(node_state + branch_gate * obs_node_state)

        pred_len = y_mark.shape[1]
        query_time = self.abs_time_encoder(y_time).unsqueeze(2).expand(-1, -1, num_variables, -1)
        node_query = node_state.unsqueeze(1).expand(-1, pred_len, -1, -1)
        global_state = 0.5 * (patch_global[:, -1] + node_state.mean(dim=1))
        global_query = global_state.unsqueeze(1).unsqueeze(2).expand(-1, pred_len, num_variables, -1)
        horizon = self.horizon_embedding[:, :pred_len].expand(bsz, -1, num_variables, -1)
        last = last_value.unsqueeze(1).unsqueeze(-1).expand(-1, pred_len, -1, 1)
        gap = y_time.expand(-1, -1, num_variables) - last_time.unsqueeze(1)  # [B,Q,N]
        trend_delta = recent_slope.unsqueeze(1) * gap  # [B,Q,N]
        trend_input = torch.stack([trend_delta, gap], dim=-1).clamp(-10.0, 10.0)  # [B,Q,N,2]
        trend_feature = self.trend_encoder(torch.nan_to_num(trend_input))  # [B,Q,N,D]
        pred_feature = torch.cat([node_query, global_query, query_time, horizon, trend_feature, last], dim=-1)

        residual_pred = last_value.unsqueeze(1) + self.trend_scale * trend_delta + self.residual_head(pred_feature).squeeze(-1)
        direct_pred = self.direct_head(pred_feature).squeeze(-1)
        output_gate = torch.sigmoid(self.output_gate(pred_feature)).squeeze(-1)
        return output_gate * residual_pred + (1.0 - output_gate) * direct_pred


class Model(nn.Module):
    """my_train: Patch-Discrete PointNet Fourier ASTGI.

    External interface is unchanged. Output dict keeps pred/true/mask so MAE/MSE
    are computed by the ASTGI pipeline using the same masked style as t-PatchGNN.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        patch_len = getattr(configs, "patch_size", getattr(configs, "astgi_patch_len", 4))
        self.encoder = PointNetPatchASTGIEncoder(
            num_variables=self.n_vars,
            pred_len=self.pred_len,
            d_model=self.d_model,
            dropout=self.dropout,
            patch_len=patch_len,
        )

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor | None, **kwargs) -> dict:
        if x_mask is None:
            x_mask = torch.ones_like(x, dtype=torch.bool)
        y = kwargs["y"]
        y_mark = kwargs["y_mark"]
        y_mask = kwargs.get("y_mask")
        prediction = self.encoder(x, x_mark, x_mask, y_mark)  # [B,Q,N]

        f_dim = -1 if getattr(self.configs, "features", "M") == "MS" else 0
        return {
            "pred": prediction[:, :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:] if y_mask is not None else None,
        }
