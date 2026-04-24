import numpy as np
from scipy.special import erf
import scipy.integrate as scpint
from scipy.interpolate import interp1d

param = 5.5

def f(x):
    """
    f(x) is e^(x^2) * (1 + erf(x))
    """
    is_scalar = np.isscalar(x) or getattr(x, 'ndim', 0) == 0
    x = np.atleast_1d(x)
    res = np.zeros_like(x, dtype=float)
    
    # Case 1: x > param (使用大值渐近近似: 1+erf(x) -> 2)
    mask_large_pos = (x > param)
    if np.any(mask_large_pos):
        val = x[mask_large_pos]
        res[mask_large_pos] = 2.0 * np.exp(val**2)

    # Case 2: x >= -param (使用标准公式)
    mask_mid = (~mask_large_pos) & (x >= -param)
    if np.any(mask_mid):
        val = x[mask_mid]
        res[mask_mid] = np.exp(val**2) * (1 + erf(val))
        
    # Case 3: x < -param (使用泰勒展开近似)
    mask_small = (x < -param)
    if np.any(mask_small):
        val = x[mask_small]
        val2 = val * val
        res[mask_small] = -1 / (np.sqrt(np.pi) * val) * (1.0 - 0.5 / val2 + 0.75 / (val2 * val2))

    return res[0] if is_scalar else res


def integrale_vec(min_arr, max_arr):
    min_arr = np.atleast_1d(min_arr)
    max_arr = np.atleast_1d(max_arr)
    res = np.zeros_like(max_arr, dtype=float)

    mask_normal = (max_arr <= param)
    for i in np.where(mask_normal)[0]:
        res[i] = scpint.quad(f, min_arr[i], max_arr[i], epsabs=1e-4, epsrel=1e-4, limit=100)[0]

    mask_large = ~mask_normal
    for i in np.where(mask_large)[0]:
        m = max_arr[i]
        
        if m > 26.0:
            res[i] = np.inf
            continue
            
        capped_min = min(min_arr[i], param)
        numerical_part = scpint.quad(f, capped_min, param, epsabs=1e-4, epsrel=1e-4, limit=100)[0]
        asymptotic_part = np.exp(m**2) / m - np.exp(param**2) / param
        
        if min_arr[i] >= param:
            res[i] = np.exp(m**2) / m - np.exp(min_arr[i]**2) / min_arr[i]
        else:
            res[i] = numerical_part + asymptotic_part

    return res

def comp_phi_tab(mu, tau, cfg):
    """
    calculate the firing rate phi(mu) with the Siegert formula:
    parameters:
        mu: input current (mV)
        tau: membrane time constant (ms)
    return:
        phi: firing rate (Hz)
    """
    if cfg.sigma_t <= 0:
        raise ValueError("Physiological Error: noise intensity sigma_t must be positive.")
    if tau <= 0 or cfg.tau_rp <= 0:
        raise ValueError("Physiological Error: membrane time constant tau and absolute refractory period tau_rp must be positive.")
    if cfg.theta <= cfg.V_r:
        raise ValueError("Physiological Error: action potential threshold theta must be greater than reset potential V_r.")
    # ensure input is array
    mu = np.atleast_1d(mu) 

    if np.any(np.abs(mu) > 100):
        raise ValueError("Physiological Error: input current mu is too large (>100), exceeding the reasonable physiological range of typical visual cortex neurons.")
    
    min_u = (cfg.V_r - mu) / cfg.sigma_t
    max_u = (cfg.theta - mu) / cfg.sigma_t
    
    integral_vals = integrale_vec(min_u, max_u)
    
    # Siegert formula
    r = 1.0 / (cfg.tau_rp + tau * np.sqrt(np.pi) * integral_vals)
    
    # if input was scalar, return scalar
    if r.size == 1:
        return r.item()
    return r

def tabulate_response(cfg):
    mu_tab = np.linspace(-cfg.mu_tab_max, cfg.mu_tab_max, int(1000*cfg.mu_tab_max))

    phi_tab_E = comp_phi_tab(mu_tab, cfg.tau_E, cfg)
    phi_tab_I = comp_phi_tab(mu_tab, cfg.tau_I, cfg)

    phi_int_E = interp1d(mu_tab, phi_tab_E, kind='linear', bounds_error=False, fill_value=(0.0, phi_tab_E[-1]))
    phi_int_I = interp1d(mu_tab, phi_tab_I, kind='linear', bounds_error=False, fill_value=(0.0, phi_tab_I[-1]))

    return phi_int_E, phi_int_I