"""
Utility functions for vegetation_grass.
Contains long, technical, or reusable logic only.
"""

import numpy as np
import logging
from numba import njit

from aeolis.utils import *

logger = logging.getLogger(__name__)

def ensure_grass_parameters(p):
    """
    Check the length of grass parameters and make iterable over species.
    """ 

    ns = p['nspecies'] # based on length of 'species_names'

    param_lists = [
        'd_tiller', 'r_stem', 'alpha_uw', 'alpha_Nt', 'alpha_0',
        'G_h', 'G_c', 'G_s', 'Hveg', 'phi_h',
        'Nt_max', 'R_cov',
        'lmax_c', 'mu_c', 'alpha_s', 'nu_s',
        'gamma_h', 'gamma_c', 'gamma_s',
        'dzb_opt_h', 'dzb_opt_c', 'dzb_opt_s',
        'beta_veg', 'm_veg', 'c1_okin', 'bounce'
        ]

    for param in param_lists:
        if param in p:
            p[param] = makeiterable(p[param]) # Coverts to array

            # If the parameter length does not match nspecies, extend it
            if len(p[param]) != ns:
                if len(p[param]) == 1:
                    p[param] = np.full(ns, p[param][0]) 
                    logger.info(f"Parameter '{param}' extended to match nspecies={ns}.")
                else:
                    logger.error(f"Parameter '{param}' length does not match nspecies={ns}.")
                    raise ValueError(f"Parameter '{param}' length does not match nspecies={ns}.")
                
        else:
            logger.error(f"Parameter '{param}' is missing for grass model.")
            raise KeyError(f"Parameter '{param}' is missing for grass model.")
            
    return p
            

def generate_grass_subgrid(x, y, veg_res_factor):
    """
    Generate a refined vegetation subgrid inside each main grid cell.
    """

    f = veg_res_factor

    # Local grid vectors (rotation-safe)
    ex_x, ex_y = np.gradient(x, axis=1), np.gradient(y, axis=1)
    ey_x, ey_y = np.gradient(x, axis=0), np.gradient(y, axis=0)

    # Subcell offsets inside parent cell
    offs = (np.arange(f) + 0.5) / f - 0.5

    # Expand base grid
    x_veg = np.repeat(np.repeat(x, f, axis=1), f, axis=0)
    y_veg = np.repeat(np.repeat(y, f, axis=1), f, axis=0)

    for j in range(f):
        for i in range(f):
            x_veg[j::f, i::f] += offs[i] * ex_x + offs[j] * ey_x
            y_veg[j::f, i::f] += offs[i] * ex_y + offs[j] * ey_y

    return x_veg, y_veg


def expand_to_subgrid(A, f):
    """Expand (ns, ny, nx) → (ns, ny*f, nx*f)."""
    return np.repeat(np.repeat(A, f, axis=1), f, axis=2)


def aggregate_from_subgrid(A, f):
    """Aggregate (ns, ny*f, nx*f) → (ns, ny, nx) by averaging."""
    ns, nyf, nxf = A.shape
    ny, nx = nyf // f, nxf // f
    return A.reshape(ns, ny, f, nx, f).mean(axis=(2,4))


def smooth_burial(s, p):
    """
    Compute smoothed burial rate over a trailing window T_burial.
    Stores bed levels over time and computes:
        dzb_veg = (zb(t) - zb(t - T_burial)) / T_burial   [m/s]
    """

    t = p['_time']
    T = p['T_burial']

    # Initialize history on first call
    if 'zb_hist' not in s:
        s['zb_hist'] = [(t, s['zb'].copy())]
        return np.zeros_like(s['zb'])

    # Append current state
    s['zb_hist'].append((t, s['zb'].copy()))

    # Drop states older than window
    t_min = t - T
    while s['zb_hist'][0][0] < t_min:
        s['zb_hist'].pop(0)
        
    t0, zb0 = s['zb_hist'][0] # Oldest retained bed level

    # Burial rate [m/s]
    return (s['zb'] - zb0) / max(t - t0, 1e-12)


@njit
def neighbourhood_average(Nt, R_cov, dx):
    """
    Compute neighbourhood average of Nt within radius R_cov.
    """

    # Determine radius in grid cells
    r = int(np.ceil(R_cov / dx))

    # Loop over grid and compute average
    Nt_avg = np.zeros_like(Nt)
    for i in range(Nt.shape[0]):
        for j in range(Nt.shape[1]):
            ssum = 0.0
            cnt = 0 

            # Define neighbourhood bounds
            i0 = max(i - r, 0)
            i1 = min(i + r + 1, Nt.shape[0])
            j0 = max(j - r, 0)
            j1 = min(j + r + 1, Nt.shape[1])

            # Loop over neighbourhood
            for ii in range(i0, i1):
                for jj in range(j0, j1):
                    if ((ii - i)**2 + (jj - j)**2) * dx**2 <= R_cov**2:
                        ssum += Nt[ii, jj]
                        cnt += 1

            # Compute average
            if cnt > 0:
                Nt_avg[i, j] = ssum / cnt

    return Nt_avg


def build_clonal_kernel(s_min, s_max, mu, dx):
    """ 
    Construct clonal dispersal kernel using a truncated Pareto distribution.
    """

    # Determine kernel radius in grid cells
    r = int(s_max / dx)

    # Create distance grid in physical space
    offsets = np.arange(-r, r + 1) * dx
    KX, KY = np.meshgrid(offsets, offsets)
    dist = np.sqrt(KX**2 + KY**2)

    # Set centre cell distance for internal recruitment
    dist[r, r] = dx / np.sqrt(6)

    # Initialise kernel
    kernel = np.zeros_like(dist)

    # Apply truncated Pareto distribution
    mask = (dist >= s_min) & (dist <= s_max)
    kernel[mask] = (
        mu
        / (s_min**(1 - mu) - s_max**(1 - mu))
        * dist[mask] ** (-mu)
    )

    # Normalize kernel
    kernel /= kernel.sum()

    return kernel, r


@njit
def apply_clonal_kernel(S_c, kernel):
    """
    Apply clonal dispersal kernel to clonal production rate S_c.
    """

    # Kernel radius
    r = kernel.shape[0] // 2

    # Initialize output
    dNt_clonal = np.zeros_like(S_c)

    # Loop over grid cells
    for i in range(S_c.shape[0]):
        for j in range(S_c.shape[1]):
            ssum = 0.0

            # Loop over kernel
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    ii = i + ki - r
                    jj = j + kj - r

                    if 0 <= ii < S_c.shape[0] and 0 <= jj < S_c.shape[1]:
                        ssum += kernel[ki, kj] * S_c[ii, jj]

            # Assign spreaded value
            dNt_clonal[i, j] = ssum

    return dNt_clonal


@njit
def sample_seed_germination(S_s, a_s, nu_s, dx):
    """
    Sample stochastic seed germination events from seed production rate S_s.
    """

    ny, nx = S_s.shape

    # Total seed production rate
    lambda_total = S_s.sum()
    dNt_seed = np.zeros_like(S_s)

    if lambda_total <= 0.0:
        return dNt_seed

    # Number of seeds this timestep
    n_seeds = np.random.poisson(lambda_total)

    if n_seeds == 0:
        return dNt_seed

    # Flatten for weighted source selection
    flat = S_s.ravel()
    probs = flat / lambda_total

    for _ in range(n_seeds):

        # Choose source cell
        j = np.random.choice(flat.size, p=probs)
        iy = j // nx
        ix = j - iy * nx

        # Sample jump distance from 2Dt distribution
        u = np.random.rand()
        r = np.sqrt(a_s * (u ** (-1.0 / (nu_s - 1.0)) - 1.0))

        # Sample direction
        theta = 2.0 * np.pi * np.random.rand()

        # Convert to grid offsets
        dy = int(round((np.sin(theta) * r) / dx))
        dx_ = int(round((np.cos(theta) * r) / dx))

        yy = iy + dy
        xx = ix + dx_

        # 5. Check bounds
        if 0 <= yy < ny and 0 <= xx < nx:
            dNt_seed[yy, xx] += 1

    return dNt_seed
