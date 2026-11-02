import numpy as np
#from .regret import regret


def get_cl_data_from_hist(counts: np.ndarray, bin_edges: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    total = counts.sum()
    if total <= 0:
        return 0.0

    #widths = np.diff(bin_edges)
    #p_hat = counts / total
    #density_hat = np.where(widths > 0, p_hat / widths, 0.0)
    # ---- make sure numeric dtype (avoid object-array slow path) ---
    widths = np.diff(bin_edges)
    n = counts.sum()

    if n <= 0 or widths.size == 0:
        return 0.0

    p_hat = counts / n

    # ---- faster & safer than np.where(p_hat/widths) ----
    density_hat = np.divide(
        p_hat,
        widths,
        out=np.zeros_like(p_hat, dtype=np.float64),
        where=widths > 0
    )


    mask = (counts > 0) & (density_hat > 0)
    negloglike = -np.sum(np.log2(density_hat[mask]) * counts[mask])
    return float(negloglike)

