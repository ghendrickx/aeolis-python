"""
Vegetation module for dune grasses (new framework).

This module is an alternative to aeolis.vegetation without overwriting it.
It describes the local growth and spreading of dune grasses,
including their effects on shear stress and sediment transport.
"""

import numpy as np
import matplotlib.pyplot as plt
from aeolis import grass_utils as gutils

def initialize(s, p):
    """
    Initialize vegetation state variables.
    Vegetation subgrid is prognostic; main grid is diagnostic.
    """

    f = p['veg_res_factor']

    # --- Make parameters iterable over species and check length -------------
    p = gutils.ensure_grass_parameters(p)

    # --- Convert yearly to secondly rates ------------------------------------
    for param in ['G_h', 'G_c', 'G_s', 
                  'dzb_tol_h', 'dzb_tol_c', 'dzb_tol_s',
                  'dzb_opt_h', 'dzb_opt_c', 'dzb_opt_s']:
        p[param] /= (365.25 * 24.0 * 3600.0)

    # --- Read main-grid vegetation state variables --------------------------
    try:
        s['Nt'][:] = p['Nt_file'].reshape((p['ny']+1, p['nx']+1, p['nspecies']))
        s['hveg'][:] = p['hveg_file'].reshape((p['ny']+1, p['nx']+1, p['nspecies']))
    except Exception as e:
        raise RuntimeError("Shape mismatch for vegetation files.") from e
    
    # --- Initialize vegetation subgrid geometry -----------------------------
    s['x_vsub'], s['y_vsub'] = gutils.generate_grass_subgrid(s['x'], s['y'], f)

    # --- Compute resolution of vegetation subgrid (assumed uniform) ---------
    p['dx_veg'] = np.sqrt((s['x_vsub'][0, 1] - s['x_vsub'][0, 0])**2 
                          + (s['y_vsub'][0, 1] - s['y_vsub'][0, 0])**2)
    dx_main = np.sqrt((s['x'][0, 1] - s['x'][0, 0])**2 
                      + (s['y'][0, 1] - s['y'][0, 0])**2)
    print(f"Subgrid resolution reduced from {dx_main:.3f} m to {p['dx_veg']:.3f} m.")

    # --- One-time lift: main grid → vegetation subgrid ----------------------
    s['Nt_vsub']   = gutils.expand_to_subgrid(s['Nt'], f)
    s['hveg_vsub'] = gutils.expand_to_subgrid(s['hveg'], f)

    # --- Build kernel functions for spreading --------------------------------
    s['kernel_c'] = [None] * p['nspecies']
    s['radius_c'] = np.zeros(p['nspecies'], dtype=int)
    for k in range(p['nspecies']):
        s['kernel_c'][k], s['radius_c'][k] = gutils.build_clonal_kernel(
            0.001, p['lmax_c'][k], p['mu_c'][k], p['dx_veg'])

    return s, p


def update(s, p):

    """
    Main vegetation update.
    All dynamics occur on the vegetation subgrid.
    """

    # --- Time step and resolution factor ------------------------------------
    dt = p['dt_veg']
    f = p['veg_res_factor']

    # --- Burial smoothing (main grid → subgrid, diagnostic → prognostic) ----
    dzb_main = gutils.smooth_burial(s, p)
    dzb_vsub = gutils.expand_to_subgrid(dzb_main[:,:,None], f)[:,:,0]

    # --- Expand main-grid state variables to subgrid (for flooding) ---------
    zb_vsub = gutils.expand_to_subgrid(s['zb'][:,:,None], f)[:,:,0]
    TWL_vsub = gutils.expand_to_subgrid(s['TWL'][:,:,None], f)[:,:,0]

    # --- Neighbourhood-averaged densities (all species) ------------------------
    Nt_avg = np.zeros_like(s['Nt_vsub'])
    for l in range(p['nspecies']):
        Nt_avg[:, :, l] = gutils.neighbourhood_average(
            s['Nt_vsub'][:, :, l], p['R_cov'][l], p['dx_veg'])

    # --- Loop over species (subgrid physics) --------------------------------
    for k in range(p['nspecies']):

        Nt   = s['Nt_vsub'][:,:,k]
        hveg = s['hveg_vsub'][:,:,k]

        # --- Burial responses ----------------------------------------------
        B_h = - np.abs(dzb_vsub - p['dzb_opt_h'][k]) / p['dzb_tol_h'][k]                        # additive factor
        B_c = np.maximum(1.0 - np.abs(dzb_vsub - p['dzb_opt_c'][k]) / p['dzb_tol_c'][k], 0.0)   # multiplicative factor
        B_s = np.maximum(1.0 - np.abs(dzb_vsub - p['dzb_opt_s'][k]) / p['dzb_tol_s'][k], 0.0)   # multiplicative factor

        # --- Spreading ------------------------------------------------------
        dNt = spreading(k, Nt, hveg, Nt_avg, B_c, B_s, p, s)                        # [tillers/dt]
        s['Nt_vsub'][:,:,k] = np.maximum(Nt + dNt, 0.0)
        ix_vegetated = (Nt > 0.0)

        # --- Local growth ---------------------------------------------------
        dhveg = p['G_h'][k] * (1.0 - hveg / p['Hveg'][k])**p['phi_h'][k] + B_h      # [m/s]
        dhveg = dhveg * dt - hveg / np.maximum(Nt, 1e-6) * dNt                      # [m/dt]
        s['hveg_vsub'][:,:,k] = np.clip(hveg + dhveg, 0.0, p['Hveg'][k])
        s['hveg_vsub'][:,:,k][~ix_vegetated] = 0.0

        # --- Mortality ------------------------------------------------------
        if p['process_tide']: # Flooding
            ix_flooded = zb_vsub < TWL_vsub
            s['hveg_vsub'][:,:,k][ix_flooded] = 0.

        # Diseased (e.g. due to burial)
        ix_decayed = (s['hveg_vsub'][:,:,k] == 0.0)
        s['Nt_vsub'][:,:,k][ix_decayed] = 0.0

    # --- Aggregate back to main grid (diagnostic only) ----------------------
    s['Nt']       = gutils.aggregate_from_subgrid(s['Nt_vsub'], f)
    s['hveg']     = gutils.aggregate_from_subgrid(s['hveg_vsub'], f)

    # --- Vegetation bending -------------------------------------------------
    bend = np.ones_like(s['hveg'])
    for k in range(p['nspecies']):
        bend[:,:,k] = (p['r_stem'][k] + (1.0 - p['r_stem'][k])
                       * (p['alpha_uw'][k] * s['uw']
                         + p['alpha_Nt'][k] * s['Nt'][:,:,k] 
                         + p['alpha_0'][k]))
    
    # --- Main-grid vegetation metrics --------------------------------------
    s['hveg_eff'] = np.clip(s['hveg'] * bend, 0.0, s['hveg'])
    s['lamveg'] = s['Nt'] * s['hveg_eff'] * p['d_tiller']
    s['rhoveg'] = s['Nt'] * np.pi * (p['d_tiller'] / 2.0)**2

    return s


def spreading(k, Nt, hveg, Nt_avg, B_c, B_s, p, s):
    """
    Spatial redistribution of vegetation:
    clonal expansion and seed dispersal.
    """

    # --- Competition-weighted relative densities (Option B) --------------------
    comp = np.zeros_like(Nt)
    for l in range(p['nspecies']):
        comp += p['alpha_comp'][k, l] * (Nt_avg[:, :, l] / p['Nt_max'][l])

    saturation = np.maximum(1.0 - comp, 0.0)

    maturity = np.clip(hveg / p['Hveg'][k], 0.0, 1.0)

    # --- Tiller production rates --------------------------------------------
    S_c = p['G_c'][k] * Nt * B_c * maturity * saturation    # [tillers/s] clonal rate 
    S_s = p['G_s'][k] * Nt * B_s * maturity * saturation    # [tillers/s] seed rate
    
    S_c *= p['dt_veg']                                      # [tillers/dt]
    S_s *= p['dt_veg']                                      # [tillers/dt]

    # --- Clonal expansion ---------------------------------------------------
    dNt_clonal = gutils.apply_clonal_kernel(S_c, s['kernel_c'][k])
    Nt_clonal_new = np.random.poisson(dNt_clonal)

    # --- Seed dispersal -----------------------------------------------------
    Nt_seed_new = gutils.sample_seed_germination(S_s, p['alpha_s'][k], 
                                                 p['nu_s'][k], p['dx_veg'])

    # --- Sum contributions --------------------------------------------------
    dNt = Nt_clonal_new + Nt_seed_new

    return dNt  


def compute_shear_reduction(s, p):
    """
    Compute vegetation-induced shear reduction.
    """

    # --- Weights for normalization -----------------------------------------
    w_sum = np.zeros_like(s['x'])
    w_num_R0 = np.zeros_like(s['x'])
    w_num_R  = np.zeros_like(s['x'])

    # --- Wind convention ID (Numba) -----------------------------------------
    if p['wind_convention'] == 'nautical':
        udir_id = 0
    elif p['wind_convention'] == 'cartesian':
        udir_id = 1
    else:
        raise ValueError(f"Unknown wind_convention: {p['wind_convention']}")

    # --- Loop over species: compute per-species local and leeside reduction --
    for k in range(p['nspecies']):

        # Weighting function based on maturity and density
        maturity = s['hveg'][:, :, k] / p['Hveg'][k]
        density  = s['Nt'][:, :, k] / p['Nt_max'][k]
        w = maturity * density

        # Local Raupach reduction per species (no weighting / no normalization)
        R0_k = 1.0 / np.sqrt(1.0 + p['m_veg'][k] * p['beta_veg'][k] * s['lamveg'][:, :, k])

        # Leeside Okin reduction per species (optional)
        if p['process_vegetation_leeside']:
            R_k = gutils.compute_okin_reduction(
                s['x'], s['y'], R0_k, s['udir'], s['hveg_eff'][:, :, k], p['c1_okin'][k], udir_id)
        else:
            R_k = R0_k

        # Accumulate weighted numerators for later normalization
        w_sum   += w
        w_num_R0 += w * R0_k
        w_num_R  += w * R_k

    # --- Final normalization (separate loop / block) ------------------------
    s['R0veg'] = np.ones_like(s['x'])
    s['Rveg']  = np.ones_like(s['x'])

    ix = w_sum != 0.0
    s['R0veg'][ix] = w_num_R0[ix] / w_sum[ix]
    s['Rveg'][ix]  = w_num_R[ix]  / w_sum[ix]

    return s



def apply_shear_reduction(s, p):
    """
    Apply vegetation-induced shear reduction to wind shear.
    """

    ets = np.zeros(s['zb'].shape)
    etn = np.zeros(s['zb'].shape)

    ix = s['ustar'] != 0

    ets[ix] = s['ustars'][ix] / s['ustar'][ix]
    etn[ix] = s['ustarn'][ix] / s['ustar'][ix]

    s['ustar'] *= s['Rveg']
    s['ustars'] = s['ustar'] * ets
    s['ustarn'] = s['ustar'] * etn

    return s


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
