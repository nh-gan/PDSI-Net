import numpy as np


def _as_mask(mask, true):
    if mask is None:
        return np.ones_like(true, dtype=bool)
    return np.asarray(mask).astype(bool)


def MAE(pred, true, mask=None):
    pred, true = np.asarray(pred), np.asarray(true)
    mask = _as_mask(mask, true)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(pred - true)[mask].mean())


def MSE(pred, true, mask=None):
    pred, true = np.asarray(pred), np.asarray(true)
    mask = _as_mask(mask, true)
    if mask.sum() == 0:
        return 0.0
    return float(((pred - true) ** 2)[mask].mean())


def RMSE(pred, true, mask=None):
    return float(np.sqrt(MSE(pred, true, mask)))


def per_variable_masked_metrics(pred, true, mask=None):
    """Unified paper protocol.

    pred/true/mask: [B,L,N]. Metrics are first computed on each variable's
    observed target positions, then averaged across variables. This prevents
    variables with more observations from dominating the final score.
    """
    pred, true = np.asarray(pred), np.asarray(true)
    mask = _as_mask(mask, true)
    if pred.ndim < 3:
        return MAE(pred, true, mask), MSE(pred, true, mask), RMSE(pred, true, mask), 1

    mae_list, mse_list = [], []
    for n in range(pred.shape[-1]):
        valid = mask[..., n]
        if valid.sum() == 0:
            continue
        err = pred[..., n] - true[..., n]
        mae_list.append(float(np.abs(err)[valid].mean()))
        mse_list.append(float((err ** 2)[valid].mean()))

    if not mae_list:
        return 0.0, 0.0, 0.0, 0
    mae = float(np.mean(mae_list))
    mse = float(np.mean(mse_list))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse, len(mae_list)


def metric(pred: np.ndarray, y: np.ndarray, y_mask: np.ndarray = None, **kwargs):
    mae, mse, rmse, n_vars = per_variable_masked_metrics(pred, y, y_mask)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAE_global": MAE(pred, y, y_mask),
        "MSE_global": MSE(pred, y, y_mask),
        "RMSE_global": RMSE(pred, y, y_mask),
        "num_eval": int(_as_mask(y_mask, y).sum()),
        "num_variables_eval": int(n_vars),
        "protocol": "per-variable masked average on normalized model output space",
    }
