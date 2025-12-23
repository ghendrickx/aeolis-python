def compute_zeta(s, p):
    """
    Compute bed–interaction factor zeta.
    """

    # Compute k_str and lambda_str here....
    lam = 1
    k = 1

    # --- Weibull function for zeta ------------------------------------------
    s['zeta'] = 1.0 - np.exp(-(s['hveg_eff'] / lam)**k)
    s['zeta'] = s['zeta'] * (1.0 - p['bounce'])
