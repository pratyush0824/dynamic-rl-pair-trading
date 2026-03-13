import numpy as np
import statsmodels.api as sm

def compute_regression_spread(price_base, price_quote):
    """
    Spread defined in the paper:
    p_i = beta0 + beta1 * p_j + s
    Spread = regression residual
    """

    x = sm.add_constant(price_quote)
    model = sm.OLS(price_base, x).fit()

    beta0 = model.params[0]
    beta1 = model.params[1]

    spread = price_base - (beta0 + beta1 * price_quote)

    return spread, beta0, beta1

def compute_zone(z, open_th=1.8, close_th=0.4):
    """
    Zones defined in paper
    """

    if z > open_th:
        return 2          # short zone
    elif z > close_th:
        return 1          # neutral short
    elif z >= -close_th:
        return 0          # close zone
    elif z >= -open_th:
        return -1         # neutral long
    else:
        return -2         # long zone