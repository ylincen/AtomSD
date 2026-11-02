import numpy as np

def dkl_density(p_train_subgroup, p_train, y_test_subgroup, edges_all, edges_subgroup):
    bin_ind_all = np.digitize(y_test_subgroup, bins=edges_all[1:-1])
    bin_ind_subgroup = np.digitize(y_test_subgroup, bins=edges_subgroup[1:-1])
    
    p_per_instance = p_train[bin_ind_all]
    p_subgroup_per_instance = p_train_subgroup[bin_ind_subgroup]
    kl_test = np.mean(np.log(p_subgroup_per_instance) - np.log(p_per_instance))
    return kl_test

def score_subgroups(y, y_train, idx, split: str, y_lim, edges_all=None, edges_subgroup=None, p_s_train=None, p_train=None, gamma=1.0):
    """
    y: The target variable (y_test or the same y_train).
    y_train: The training target variable.
    idx: subgroup indices.
    split: The data split to score ("train" or "test").
    gamma: Size correction exponent.
    """
    y = np.asarray(y)
    n = len(y)

    # Build subgroup mask
    # Assuming idx is a boolean mask (Syflow, RSD, SD_mu, SD_KL)
    if isinstance(idx, (list, tuple, np.ndarray)) and np.asarray(idx).dtype == bool:
        mask = np.asarray(idx)
        if len(mask) != n:
            raise ValueError("Boolean mask must match y length.")
    else:
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(idx, dtype=int)] = True

    y_s = y[mask]
    n_s = len(y_s)

    if n_s == 0:
        raise ValueError("Subgroup is empty.")

    # Edges `learned' from training data
    if edges_all is None:
        assert edges_subgroup is None
        edges_all = np.histogram_bin_edges(y_train, bins="fd", range=(y_lim[0], y_lim[1]))
        edges_subgroup = np.histogram_bin_edges(y_s, bins="fd", range=(y_lim[0], y_lim[1]))

    # Histograms
    c_all, _ = np.histogram(y, bins=edges_all)
    c_prior, _ = np.histogram(y, bins = edges_subgroup)
    c_s, _   = np.histogram(y_s, bins=edges_subgroup)

    # Add a pseudo-count (Laplace correction for all and bayes correction for subgroup)
    c_all = c_all + 1
    c_prior = c_prior + 1
    bayes_correction = c_prior / c_prior.sum() * (len(edges_subgroup) - 1) # so the \alpha_0 corresponds to the Conjugate prior is equal to the number of bins
    c_s = c_s + bayes_correction

    if p_s_train is None and p_train is None:
        # Convert to density
        p = c_all / c_all.sum() / (edges_all[1:] - edges_all[:-1])
        p_s = c_s / c_s.sum() / (edges_subgroup[1:] - edges_subgroup[:-1])
    else:
        p = p_train
        p_s = p_s_train

    kl = dkl_density(p_s, p, y_s, edges_all, edges_subgroup)
    # -----------------------------
    # Size correction
    # -----------------------------
    support = (n_s / n) ** gamma
    kl_corr = support * kl

    return {
        f"KL_{split}": float(kl),
        f"support_{split}": n_s,
        f"support_ratio_{split}": float(n_s / n),
        f"KL_corrected_{split}": float(kl_corr),
    }, edges_all, edges_subgroup, p_s, p