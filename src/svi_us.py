"""Global eSSVI surface fitting for short-dated US single-stock options.

Replaces the TXO-era Model 1 (eSSVI with a FROZEN rho_0 = -0.95 plus a 5-network
additive ensemble). Three separate things in that design are falsified on this
data:

1. rho_0 = -0.95 implies a wing-slope ratio (1-rho)/(1+rho) of 39. The measured
   median wing ratio across all eight symbols is ~2.0, per-slice fitted rho runs
   -0.52 (SPY) to -0.12 (TSLA), and fewer than 2.1% of slices anywhere are
   compatible with rho < -0.90. It was calibrated to a TAIEX INDEX left skew;
   single names — especially high-beta NVDA/TSLA — do not have it.
2. The exponential rho(tau) decay is three parameters fitted to 3-4 expiries
   spanning 10-67 days: unidentifiable. eSSVI already allows one free rho per
   slice, which is both more flexible and actually identified.
3. Interpolating theta_t smoothly in calendar tau is what produced the
   "penalty toxicity". True theta_t carries a STEP of sigma_j^2 at each earnings
   announcement; a smooth interpolant crossing that step overshoots and
   manufactures negative dw/dtau in the FIT (the quotes themselves are
   calendar-monotone at the money in 100% of adjacent-expiry pairs).

Design:
- Fit the DE-EVENTED total variance w_diff = w_obs - n_events * sigma_j^2, so
  the term structure is smooth and theta is genuinely monotone.
- Parameterize so that no-arbitrage holds BY CONSTRUCTION over an unconstrained
  box, which removes the degenerate w -> 0 optimum entirely (there is no penalty
  to trade off against, and w = 0 is simply not a better fit):
    theta_1 = softplus(v_1),  theta_i = theta_{i-1} + softplus(v_i)   [calendar]
    rho_i   = tanh(u_i)
    psi_i   = 4 / (1 + |rho_i|) * sigmoid(z_i)                        [butterfly]
  The psi map enforces Gatheral-Jacquier's psi(1+|rho|) < 4 exactly.
- 3N parameters (N = 3-4 slices) against ~29 quotes per slice; solved with
  scipy least_squares in milliseconds on CPU, no GPU and no penalty weights.
"""
import numpy as np
from scipy.optimize import least_squares

SQRT_EPS = 1e-12


# ── Parameterization ──────────────────────────────────────────────────

def _softplus(x):
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def unpack(params, n_slices):
    """Unconstrained vector -> (theta, rho, psi).

    BUTTERFLY is guaranteed by construction, via BOTH Gatheral-Jacquier
    sufficient conditions (Thm 4.2), not just the first:
        (i)  theta*phi*(1+|rho|)  < 4   <=>  psi(1+|rho|) < 4
        (ii) theta*phi^2*(1+|rho|) <= 4  <=>  psi <= 2*sqrt(theta/(1+|rho|))
    An earlier version enforced only (i). Condition (ii) binds precisely when
    theta is small and psi moderate — i.e. at short maturities, which is the
    entire dataset — and its omission let the risk-neutral density g(k) go
    NEGATIVE (measured min density -2.7 on SPY) while the gate still reported
    "butterfly ok". psi is therefore capped by the MINIMUM of the two bounds.

    ATM CALENDAR is guaranteed by construction: theta is a cumulative sum of
    strictly positive increments, so theta is strictly increasing (the 1e-6
    floor matters — softplus underflows to exactly 0.0 for very negative
    inputs, which would make theta merely non-decreasing).

    FULL-SMILE CALENDAR (w_i(k) <= w_{i+1}(k) for all k) is NOT implied by
    theta monotonicity alone once each slice has a free rho and psi. It is
    handled by the deterministic projection in `enforce_calendar` and then
    VERIFIED numerically — not asserted from a theorem.

    psi is left FREE per slice (subject only to the butterfly bound). An
    earlier version forced psi to be increasing across slices to suppress
    crossings; that silently removed decreasing-psi surfaces from the family
    altogether and cost accuracy, so the restriction was dropped in favour of
    the projection.
    """
    v = params[:n_slices]
    u = params[n_slices:2 * n_slices]
    z = params[2 * n_slices:3 * n_slices]
    theta = np.cumsum(_softplus(v) + 1e-6)
    rho = np.tanh(u)
    one_p = 1.0 + np.abs(rho)
    psi_max = np.minimum(4.0 / one_p, 2.0 * np.sqrt(theta / one_p))
    psi = psi_max * np.clip(_sigmoid(z), 1e-6, 0.999)
    return theta, rho, psi


def essvi_w(k, theta, rho, psi):
    """eSSVI total variance for one slice.

    w(k) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)),
    with phi = psi / theta.
    """
    theta = max(float(theta), 1e-10)
    phi = psi / theta
    x = phi * np.asarray(k, dtype=float)
    return 0.5 * theta * (1.0 + rho * x
                          + np.sqrt(np.maximum((x + rho) ** 2 + 1.0 - rho ** 2,
                                               SQRT_EPS)))


# ── Fitting ───────────────────────────────────────────────────────────

def _residuals(params, slices, n_slices):
    theta, rho, psi = unpack(params, n_slices)
    out = []
    for i, s in enumerate(slices):
        w_hat = essvi_w(s['k'], theta[i], rho[i], psi[i])
        out.append(s['weight'] * (w_hat - s['w']))
    return np.concatenate(out)


def _init(slices):
    n = len(slices)
    theta0 = np.array([max(s['w'][np.argmin(np.abs(s['k']))], 1e-6)
                       for s in slices], dtype=float)
    theta0 = np.maximum.accumulate(theta0)
    inc = np.diff(np.concatenate([[0.0], theta0]))
    v = np.log(np.expm1(np.maximum(inc, 1e-6)))
    u = np.full(n, np.arctanh(-0.25))       # mild left skew, single-name-like
    z = np.zeros(n)                          # psi at half its admissible range
    return np.concatenate([v, u, z])


def enforce_calendar(fit, k_grid=None, max_passes=60):
    """Remove any residual smile-wide calendar crossing by minimally raising
    theta on the longer slice.

    w(k) is increasing in theta at fixed (rho, psi) — the eSSVI slice is
    theta/2 * (1 + ...) with phi = psi/theta, and raising theta lifts the whole
    curve — so this always converges, and it is a deterministic projection
    rather than a penalty with a weight to tune.
    """
    if k_grid is None:
        k_grid = np.linspace(-0.6, 0.6, 121)
    theta = fit['theta'].copy()
    rho, psi = fit['rho'], fit['psi']
    for _ in range(max_passes):
        worst = 0.0
        for i in range(len(theta) - 1):
            w0 = essvi_w(k_grid, theta[i], rho[i], psi[i])
            w1 = essvi_w(k_grid, theta[i + 1], rho[i + 1], psi[i + 1])
            gap = float(np.max(w0 - w1))
            if gap > 0:
                # Geometric escalation: a fixed gap-sized bump can converge
                # arbitrarily slowly, because raising theta also shrinks
                # phi = psi/theta and reshapes the slice. Guarantee progress.
                theta[i + 1] += max(gap * 1.1, theta[i + 1] * 0.02, 1e-9)
                worst = max(worst, gap)
        if worst == 0.0:
            break
    out = dict(fit)
    out['theta'] = theta
    return out


def fit_snapshot(slices, max_nfev=400):
    """Fit Global eSSVI to one (ticker, date).

    Args:
        slices: list (ordered by maturity) of dicts with
            'k'      log-moneyness array
            'w'      DE-EVENTED total variance array
            'tau'    maturity (years)
            'weight' per-quote weights (e.g. inverse half-spread)
    Returns dict with theta/rho/psi arrays, rmse, and the raw solution.
    """
    n = len(slices)
    x0 = _init(slices)
    sol = least_squares(_residuals, x0, args=(slices, n), method='trf',
                        max_nfev=max_nfev)
    theta, rho, psi = unpack(sol.x, n)
    resid = _residuals(sol.x, slices, n)
    fit = {'theta': theta, 'rho': rho, 'psi': psi,
           'tau': np.array([s['tau'] for s in slices]),
           'rmse_w': float(np.sqrt(np.mean(resid ** 2))),
           'params': sol.x, 'success': bool(sol.success)}
    return enforce_calendar(fit)


def surface_w(fit, k, slice_idx):
    return essvi_w(k, fit['theta'][slice_idx], fit['rho'][slice_idx],
                   fit['psi'][slice_idx])


# ── Diagnostics ───────────────────────────────────────────────────────

def iv_rmse(fit, slices):
    """RMSE in implied-vol points (the units the bid-ask noise floor is in)."""
    errs = []
    for i, s in enumerate(slices):
        w_hat = np.maximum(surface_w(fit, s['k'], i), 1e-12)
        iv_hat = np.sqrt(w_hat / s['tau'])
        iv_obs = np.sqrt(np.maximum(s['w'], 1e-12) / s['tau'])
        errs.append(iv_hat - iv_obs)
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


def cv_iv_rmse(slices, n_folds=4):
    """Strike-axis cross-validated IV RMSE — the honest generalization number.

    Every fold holds out a strided subset of strikes from EVERY slice, so the
    model must interpolate across the smile rather than memorize quotes.
    """
    errs = []
    for f in range(n_folds):
        tr, te = [], []
        ok = True
        for s in slices:
            m = np.arange(len(s['k'])) % n_folds != f
            if m.sum() < 5 or (~m).sum() < 1:
                ok = False
                break
            tr.append({'k': s['k'][m], 'w': s['w'][m], 'tau': s['tau'],
                       'weight': s['weight'][m]})
            te.append({'k': s['k'][~m], 'w': s['w'][~m], 'tau': s['tau'],
                       'weight': s['weight'][~m]})
        if not ok:
            continue
        fit = fit_snapshot(tr)
        for i, s in enumerate(te):
            w_hat = np.maximum(surface_w(fit, s['k'], i), 1e-12)
            errs.append(np.sqrt(w_hat / s['tau'])
                        - np.sqrt(np.maximum(s['w'], 1e-12) / s['tau']))
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else np.nan


def arbitrage_report(fit, k_grid=None):
    """Verify the constraints the parameterization is supposed to guarantee.

    Returns butterfly slack (LHS of psi(1+|rho|) < 4), whether theta is
    strictly increasing, and the calendar violation rate on a strike grid.
    """
    if k_grid is None:
        k_grid = np.linspace(-0.6, 0.6, 121)
    theta, rho, psi = fit['theta'], fit['rho'], fit['psi']
    one_p = 1.0 + np.abs(rho)
    lhs1 = psi * one_p                       # GJ condition (i)
    lhs2 = psi ** 2 * one_p / np.maximum(theta, 1e-12)   # GJ condition (ii)

    # The parameter conditions are only SUFFICIENT proxies. What actually
    # matters is the risk-neutral density itself, so measure it directly —
    # an earlier version reported "butterfly ok" from condition (i) alone
    # while the density was negative.
    dens = np.stack([_density(k_grid, theta[i], rho[i], psi[i])
                     for i in range(len(theta))])
    W = np.stack([essvi_w(k_grid, theta[i], rho[i], psi[i])
                  for i in range(len(theta))])
    cal_viol = float(np.mean(np.diff(W, axis=0) < 0)) if len(theta) > 1 else 0.0
    return {'butterfly_lhs_max': float(np.max(lhs1)),
            'butterfly_lhs2_max': float(np.max(lhs2)),
            'min_density': float(np.min(dens)),
            'butterfly_ok': bool(np.min(dens) > 0),
            'theta_increasing': bool(np.all(np.diff(theta) > 0)),
            'calendar_violation_rate': cal_viol,
            'rho_min': float(np.min(rho)), 'rho_max': float(np.max(rho)),
            'wing_ratio': float(np.median((1 - rho) / (1 + rho)))}


def _density(k, theta, rho, psi):
    """Gatheral's g(k): the risk-neutral density factor / Dupire denominator."""
    theta = max(float(theta), 1e-12)
    phi = psi / theta
    x = phi * np.asarray(k, dtype=float)
    D = np.sqrt(np.maximum((x + rho) ** 2 + 1.0 - rho ** 2, 1e-16))
    w = 0.5 * theta * (1.0 + rho * x + D)
    dw = 0.5 * theta * phi * (rho + (x + rho) / D)
    d2w = 0.5 * theta * phi ** 2 * (1.0 - rho ** 2) / D ** 3
    w = np.maximum(w, 1e-12)
    return (1.0 - (k / w) * dw
            + 0.25 * (-0.25 - 1.0 / w + k ** 2 / w ** 2) * dw ** 2
            + 0.5 * d2w)
