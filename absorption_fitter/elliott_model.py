# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 22:03:47 2026

@author: sasuk
"""

# -*- coding: utf-8 -*-
"""
elliott_model.py
================
Elliott formula 吸收光譜模型 (支援 2D 與 3D 材料，含 sech 展寬)
"""

import numpy as np

EPS = 1e-8

def sech(x):
    """數值穩定版本的 sech(x) = 1/cosh(x)，避免 overflow。"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    big = np.abs(x) > 30
    out[big] = 0.0
    out[~big] = 1.0 / np.cosh(x[~big])
    return out

def exciton_term_raw(E, E0, Eg, Gamma_ex, n_max=8, dim="2D"):
    """
    激子吸收項 (未乘上 alpha0)
    支援 2D/Quasi-2D 與 3D Bulk 兩種模型。
    """
    E = np.asarray(E, dtype=float)
    total = np.zeros_like(E)
    for n in range(1, int(n_max) + 1):
        if dim == "2D":
            m = n - 0.5
            Eb_n = E0 / (m ** 2)
            coeff = (4.0 * E0) / (m ** 3)
        else: # "3D"
            m = float(n)
            Eb_n = E0 / (m ** 2)
            coeff = (2.0 * E0) / (m ** 3)
            
        total += coeff * sech((E - Eg + Eb_n) / Gamma_ex)
    return total

def _continuum_factor(x, Eg, E0, dim="2D"):
    """
    將 JDOS 與 Sommerfeld factor 結合的連續態有效因子。
    注意：x 是連續態內部的能階 (x >= Eg)。
    """
    x = np.asarray(x, dtype=float)
    denom = x - Eg
    safe_denom = np.where(denom < EPS, EPS, denom)
    sqrt_term = np.sqrt(np.maximum(E0, 0.0) / safe_denom)
    
    if dim == "2D":
        # 2D JDOS 為常數，只剩 Sommerfeld factor
        return 2.0 / (1.0 + np.exp(-2.0 * np.pi * sqrt_term))
    else:
        # 3D JDOS ∝ sqrt(x-Eg)，與 Sommerfeld 乘積化簡後：
        # 避免極高能下 exp(-0) -> 1 造成分母為 0，加上 EPS 保護
        exp_term = np.exp(-2.0 * np.pi * sqrt_term)
        num = 2.0 * np.pi * np.sqrt(np.maximum(E0, 0.0))
        return num / (1.0 - exp_term + EPS)

def continuum_term_raw(E_array, Eg, E0, Gamma_c, b, n_max=8, x_upper=10.0, n_grid=400, dim="2D"):
    """
    連續態吸收項，對每個 E 數值積分 (JDOS*F(x) 置於積分內):
        ∫_{Eg}^{Eg+x_upper} sech((E-x)/Gamma_c) * (JDOS * F(x)) * 1/(1-b(x-Eg)) dx
    """
    E_array = np.asarray(E_array, dtype=float)
    x = np.linspace(Eg, Eg + x_upper, n_grid)

    denom = 1.0 - b * (x - Eg)
    denom = np.where(np.abs(denom) < EPS,
                      np.where(denom >= 0, EPS, -EPS),
                      denom)

    # 取得結合 JDOS 的 Sommerfeld 增強因子
    F_effective = _continuum_factor(x, Eg, E0, dim=dim)

    diff = E_array[:, None] - x[None, :]
    integrand_core = (sech(diff / Gamma_c) * F_effective[None, :]) / denom[None, :]
    
    return np.trapezoid(integrand_core, x, axis=1)

def total_absorption(E, alpha0, E0, Eg, Gamma_ex, Gamma_c, b, n_max=8, x_upper=10.0, dim="2D"):
    """完整模型: alpha0 * (exciton + continuum)"""
    exc = exciton_term_raw(E, E0, Eg, Gamma_ex, n_max, dim)
    cont = continuum_term_raw(E, Eg, E0, Gamma_c, b, n_max, x_upper, dim=dim)
    return alpha0 * (exc + cont)

def component_curves(E, alpha0, E0, Eg, Gamma_ex, Gamma_c, b, n_max=8, x_upper=10.0, dim="2D"):
    exc = alpha0 * exciton_term_raw(E, E0, Eg, Gamma_ex, n_max, dim)
    cont = alpha0 * continuum_term_raw(E, Eg, E0, Gamma_c, b, n_max, x_upper, dim=dim)
    total = exc + cont
    return exc, cont, total