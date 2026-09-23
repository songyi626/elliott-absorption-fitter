# -*- coding: utf-8 -*-
"""
estimator.py
============
根據目前選擇範圍內的實驗數據，自動估算 Elliott formula 六個參數的合理初始值。

策略 (皆為工程上常見的粗略估計，供使用者省去手動嘗試的麻煩，
擬合前仍建議目視檢查是否合理):

- alpha0   : 直接取資料最大值的量級
- Eg, Gamma_ex :
    先嘗試用 scipy.signal.find_peaks 找出資料中最顯著的激子吸收峰。
    若找到峰:
        峰值位置 E_peak 對應 n=1 激子態峰值 (E = Eg - E0)，故 Eg = E_peak + E0_guess
        峰的半高寬 (FWHM) 換算為 sech 函數的展寬參數 Gamma_ex
    若找不到明顯的峰 (例如吸收邊很平滑、沒有突出的激子峰):
        改用「最大斜率位置」(吸收邊上升最陡的地方) 當作 Eg 的估計
- E0       : 使用常見的預設值 (~20 meV)，此參數通常需要使用者依材料系統微調
- Gamma_c  : 假設略大於 Gamma_ex (連續態通常展寬較寬)
- b        : 給予保守的預設值 (較不敏感的高階修正項)
"""

import numpy as np
from scipy.signal import find_peaks, peak_widths


def estimate_initial_parameters(E, A, dim="2D", user_E0_guess=None):
    """
    E, A: 1D numpy array
    dim: "2D" 或 "3D"
    user_E0_guess: 若使用者有在介面上先填入 E0，則以此為基準來推算 Eg
    """
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)
    order = np.argsort(E)
    E = E[order]
    A = A[order]

    result = {}
    A_max = float(np.max(A))
    A_min = float(np.min(A))
    result["alpha0"] = max(A_max, 1e-6)

    # 如果沒有傳入使用者猜測值，才給予經驗預設值
    if user_E0_guess is not None and user_E0_guess > 0:
        E0_guess = user_E0_guess
    else:
        # 經驗預設值: 3D 約 30 meV, 2D 的 E0 通常較大 (例如 40 meV -> 束縛能 160 meV)
        E0_guess = 0.04 if dim == "2D" else 0.03  

    prominence = 0.05 * (A_max - A_min) if A_max > A_min else None
    peaks, props = find_peaks(A, prominence=prominence)

    if len(peaks) > 0:
        best_idx = peaks[int(np.argmax(props["prominences"]))]
        E_peak = E[best_idx]

        widths_result = peak_widths(A, [best_idx], rel_height=0.5)
        width_in_points = widths_result[0][0]
        dE = float(np.mean(np.diff(E))) if len(E) > 1 else 1.0
        fwhm = width_in_points * dE
        gamma_ex_guess = max(fwhm / 2.634, 0.005)

        # 核心修正：依據維度正確反推能隙 Eg
        if dim == "2D":
            result["Eg"] = float(E_peak + 4.0 * E0_guess)
        else:
            result["Eg"] = float(E_peak + E0_guess)
            
        result["Gamma_ex"] = float(gamma_ex_guess)
        result["_peak_found"] = True
    else:
        dA_dE = np.gradient(A, E)
        idx_max_slope = int(np.argmax(dA_dE))
        result["Eg"] = float(E[idx_max_slope])
        result["Gamma_ex"] = 0.02
        result["_peak_found"] = False

    result["E0"] = E0_guess
    result["Gamma_c"] = max(result["Gamma_ex"] * 1.5, 0.02)
    result["b"] = 0.1

    return result