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
    for param in ['G_h', 'G_c', 'G_s', 'dzb_opt_h', 'dzb_opt_c', 'dzb_opt_s']:
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

    # --- Loop over species (subgrid physics) --------------------------------
    for ns in range(p['nspecies']):

        Nt   = s['Nt_vsub'][:,:,ns]
        hveg = s['hveg_vsub'][:,:,ns]

        # --- Burial responses ----------------------------------------------
        B_h = p['gamma_h'][ns] * (dzb_vsub - p['dzb_opt_h'][ns])
        B_c = np.maximum(p['gamma_c'][ns] * (dzb_vsub - p['dzb_opt_c'][ns]), 0.0)
        B_s = np.maximum(p['gamma_s'][ns] * (dzb_vsub - p['dzb_opt_s'][ns]), 0.0)

        # --- Spreading ------------------------------------------------------
        dNt = spreading(ns, Nt, hveg, p, s) # [tillers/s]

        # --- Local growth ---------------------------------------------------
        dhveg = p['G_h'][ns] * (1.0 - hveg / p['Hveg'][ns])**p['phi_h'][ns] + B_h   # [m/s]
        dhveg = dhveg * dt - hveg / np.maximum(Nt, 1e-6) * dNt                      # [m/dt]

        # --- Update prognostic subgrid state --------------------------------
        s['Nt_vsub'][:,:,ns]   = np.maximum(Nt + dNt, 0.0)
        s['hveg_vsub'][:,:,ns] = np.clip(hveg + dhveg, 0.0, p['Hveg'][ns])

        # --- Mortality ------------------------------------------------------

        # Flooding
        if p['process_tide']:
            ix_flooded = zb_vsub < TWL_vsub
            s['hveg_vsub'][:,:,ns][ix_flooded] = 0.

        # Diseased (e.g. due to burial)
        ix_decayed = (s['hveg_vsub'][:,:,ns] == 0.0)
        s['Nt_vsub'][:,:,ns][ix_decayed] = 0.0

    # --- Aggregate back to main grid (diagnostic only) ----------------------
    s['Nt']       = gutils.aggregate_from_subgrid(s['Nt_vsub'], f)
    s['hveg']     = gutils.aggregate_from_subgrid(s['hveg_vsub'], f)

    # --- Vegetation bending -------------------------------------------------
    bend = np.ones_like(s['hveg'])
    for ns in range(p['nspecies']):
        bend[:,:,ns] = (p['r_stem'][ns] + 
                        (1.0 - p['r_stem'][ns])
                        * (p['alpha_uw'][ns] * s['uw']
                         + p['alpha_Nt'][ns] * s['Nt'][:,:,ns] 
                         + p['alpha_0'][ns]))
    
    # --- Main-grid vegetation metrics --------------------------------------
    s['hveg_eff'] = np.clip(s['hveg'] * bend, 0.0, s['hveg'])
    s['lamveg'] = s['Nt'] * s['hveg_eff'] * p['d_tiller']
    s['rhoveg'] = s['Nt'] * np.pi * (p['d_tiller'] / 2.0)**2


def spreading(ns, Nt, hveg, p, s):
    """
    Spatial redistribution of vegetation:
    clonal expansion and seed dispersal.
    """

    # --- Neighbourhood average density --------------------------------------
    Nt_avg = gutils.neighbourhood_average(Nt, p['R_cov'][ns], p['dx_veg'])
    saturation = np.maximum(1.0 - Nt_avg / p['Nt_max'][ns], 0.0)
    maturity = np.clip(hveg / p['Hveg'][ns], 0.0, 1.0)

    # --- Tiller production rates --------------------------------------------
    S_c = p['G_c'][ns] * Nt * maturity * saturation  # [tillers/s] clonal rate 
    S_s = p['G_s'][ns] * Nt * maturity               # [tillers/s] seed rate

    # --- Clonal expansion ---------------------------------------------------
    dNt_clonal = gutils.apply_clonal_kernel(S_c, s['kernel_c'][ns])
    Nt_clonal_new = np.random.poisson(dNt_clonal * p['dt_veg'])

    # --- Seed dispersal -----------------------------------------------------
    Nt_seed_new = gutils.sample_seed_germination(S_s, p['alpha_s'][ns], 
                                                 p['nu_s'][ns], p['dx_veg'])

    # --- Sum contributions --------------------------------------------------
    dNt = Nt_clonal_new + Nt_seed_new

    return dNt  


def compute_shear_reduction(s, p):
    """
    Compute vegetation-induced shear reduction.
    """

    s['R0veg'] = np.ones((p['ny']+1, p['nx']+1))
    R0veg = np.zeros_like(s['R0veg'])
    w_sum = np.zeros_like(s['R0veg'])

    # --- Species-weighted frontal density ----------------------------------
    for ns in range(p['nspecies']):
        maturity = s['hveg'][:,:,ns] / p['Hveg'][ns]
        density  = s['Nt'][:,:,ns] / p['Nt_max'][ns]
        w = maturity * density

        # Compute shear reduction per species
        R0veg += w * 1.0 / np.sqrt(1.0 + p['m_veg'][ns] * p['beta_veg'][ns] * s['lamveg'][:,:,ns])
        w_sum += w

    # Normalize weighted sum
    ix = w_sum != 0.0
    s['R0veg'][ix] = R0veg[ix] / w_sum[ix]

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
