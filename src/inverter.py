import nlopt
import numpy as np
import time
from joblib import delayed, Parallel
from tqdm import tqdm

from src.objective import Objective
from src.crlb import compute_precision
from src.helpers import CODES


class Inverter:
    def __init__(self,
                 phase,
                 obs,
                 disease,
                 populations,
                 nat_driver,
                 theta_upper=0.5,
                 alpha_upper=0.0,
                 k=10.0):
        """theta_upper: upper bound on pairwise coupling θ; 0.0 pins θ=0.
        alpha_upper: upper bound on national-driver mixing weight α; 0.0 pins α=0.
        nat_driver: dict {season -> array} of per-capita national infection
        rate. Required (no default). Percolates into Objective / Packer /
        compute_g / compute_precision.
        k: Negative-Binomial dispersion.
        """
        self.objective = Objective(obs=obs,
                                   phase=phase,
                                   disease=disease,
                                   populations=populations,
                                   nat_driver=nat_driver,
                                   theta_upper=theta_upper,
                                   alpha_upper=alpha_upper,
                                   k=k)

    def fit(self,
            n0,
            n_jobs=-1,
            fname=None):

        results = Parallel(n_jobs=n_jobs)(
            delayed(single_optimization)(self.objective)
            for _ in tqdm(range(n0))
        )

        theta_upper = self.objective.packer.theta_upper
        n_ok = sum(1 for r in results if r['success'])
        n_fail = n0 - n_ok
        print(f"Optimization (theta_upper={theta_upper}): "
              f"{n_ok} succeeded + {n_fail} failed / {n0} total")

        best = min(results, key=lambda r: r['fun'])
        self.x = best['x']
        self.success = best['success']
        self.fun = best['fun']
        self.nlopt_code = best['nlopt_code']
        self.desc = best['desc']
        self.runtime = best['runtime']
        # Aggregated (alpha, theta) FIM from R_g stack, plus theta CRLB.
        self.R_gs = best['R_gs']
        self.J_aa = best['J_aa']
        self.J_at = best['J_at']
        self.J_tt = best['J_tt']
        self.crlb_theta = best['crlb_theta']
        self.precision = best['precision']
        # Raw log-likelihood: objective is NLL / n_obs, so log_lik = -fun * n_obs.
        self.log_likelihood = -best['fun'] * self.objective.n_obs
        self.theta_upper = theta_upper

        if fname:
            self._plot_reconstruction(fname=fname)

        return self

    def _plot_reconstruction(self, fname=None):
        """Plot observed vs fitted incidence for each region and season."""
        import matplotlib.pyplot as plt
        from src.data_loader import t_to_date

        obj = self.objective
        packer = obj.packer
        fitted = packer.unpack(self.x)

        sim = packer.sim(fitted, obj.phase, obj.disease)
        obs = obj.obs

        regions = list(packer.regions)
        show_seasons = [2013, 2015, 2017]
        seasons = [s for s in show_seasons if s in packer.seasons] or list(packer.seasons)[:3]
        n_seasons = len(seasons)

        fig, axes = plt.subplots(len(regions), n_seasons, figsize=(4 * n_seasons, 3 * len(regions)),
                                 squeeze=False, sharey='row', sharex='col')

        for i, region in enumerate(regions):
            for j, season in enumerate(seasons):
                ax = axes[i, j]
                obs_r = obs[(obs['season'] == season) & (obs['region'] == region)].sort_values('t')
                sim_r = sim[(sim['season'] == season) & (sim['region'] == region)].sort_values('t')

                obs_dates = obs_r['t'].apply(t_to_date)
                sim_dates = sim_r['t'].apply(t_to_date)

                ax.plot(obs_dates, obs_r['incidence'], 'ko', markersize=3, label='Observed')
                ax.plot(sim_dates, sim_r['mu'] * obj.disease.rho, 'r-', linewidth=1.5, label=r'$\rho\mu$')
                ax.tick_params(axis='x', rotation=30, labelsize=7)

                if j == 0:
                    ax.set_ylabel(region)
                if i == 0:
                    ax.set_title(f'{season}')
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)

        fig.tight_layout()
        if fname:
            plt.savefig(fname + ".png", dpi=150, bbox_inches='tight')
            plt.savefig(fname + ".pdf", bbox_inches='tight')
            print(f"Saved {fname}")
        else:
            plt.show()


def single_optimization(objective):

    start = time.time()
    packer = objective.packer
    n_regions = packer.n_regions
    n_seasons = packer.n_seasons
    M = n_regions * n_seasons
    n = packer.n_params

    opt = nlopt.opt(nlopt.LD_SLSQP, n)
    opt.set_xtol_rel(1e-10)
    opt.set_maxtime(100)
    opt.set_min_objective(objective)
    for idx in range(M):
        def make_constraint(i):
            def constraint(x, grad):
                if grad.size > 0:
                    grad[:] = 0
                    grad[i] = 1
                    grad[i + M] = 1
                return x[i] + x[i + M] - 1
            return constraint
        opt.add_inequality_constraint(make_constraint(idx), 1e-8)
    opt.set_lower_bounds([0.0] * n)
    # Parameter vector layout matches Packer.pack: [S, I, α?, θ?]. Bounds
    # for α and θ are appended in that order iff their upper bound > 0.
    upper_bounds = [1.0] * (2 * M)
    if packer.alpha_upper:
        upper_bounds.append(packer.alpha_upper)
    if packer.theta_upper:
        upper_bounds.append(packer.theta_upper)
    opt.set_upper_bounds(upper_bounds)

    x0 = packer.random_vector()
    try:
        x = opt.optimize(x0)
        code = opt.last_optimize_result()
        result = dict(x=x, fun=opt.last_optimum_value(), nlopt_success=1<=code<=4, nlopt_code=code, desc=CODES[code])
    except Exception:
        fun_x0 = objective(x0)
        result = dict(x=x0, fun=fun_x0, nlopt_success=False, nlopt_code=-1, desc="failed")

    failed_seasons = []
    # Compute per-season R_g (2x2 upper-triangular trailing block of the R
    # factor from QR of sqrt(W) G, with columns [S1_0,I1_0,S2_0,I2_0,alpha,theta]).
    # Aggregation across seasons and inversion for theta CRLB are handled below.
    fitted = packer.unpack(result['x'])
    R_gs = []
    for season_idx, season in enumerate(packer.seasons):
        N = np.array([packer.populations[(season, packer.regions[i])]
                      for i in range(n_regions)])
        try:
            R_g = compute_precision(
                S0=fitted['S_init'][season_idx, :] * N,
                I0=fitted['I_init'][season_idx, :] * N,
                gamma=objective.disease.gamma,
                theta=fitted.get('theta', 0.0),
                Ts=packer.all_Ts[season],
                beta0=objective.disease.beta0,
                delta=objective.disease.delta,
                rho=objective.disease.rho,
                phase=objective.phase,
                N=N,
                I_nat_pc=packer.nat_driver[season],
                alpha=fitted.get('alpha', 0.0),
                k=objective.k
            )
            R_gs.append(R_g)
        except Exception:
            failed_seasons.append(str(season))
            R_gs.append(np.full((2, 2), np.nan))

    if failed_seasons:
        result['desc'] += f" | failed R_g: {', '.join(failed_seasons)}"

    # Aggregate: J_total = sum_s R_g(s)^T R_g(s). Profile alpha out of the
    # aggregated 2x2 to get theta CRLB via the 2x2 inverse formula:
    #   J_total = [[J_aa, J_at], [J_at, J_tt]],
    #   det     = J_aa * J_tt - J_at^2,
    #   Var(theta) = J_aa / det = 1 / (J_tt - J_at^2 / J_aa).
    J_total = np.zeros((2, 2))
    for R_g in R_gs:
        if np.all(np.isfinite(R_g)):
            J_total += R_g.T @ R_g
    J_aa, J_at, J_tt = J_total[0, 0], J_total[0, 1], J_total[1, 1]
    det = J_aa * J_tt - J_at ** 2
    if det > 0 and J_aa > 0:
        crlb_theta = J_aa / det
        precision = 1.0 / crlb_theta
    else:
        crlb_theta = np.inf
        precision = 0.0

    result['R_gs'] = R_gs
    result['J_aa'] = J_aa
    result['J_at'] = J_at
    result['J_tt'] = J_tt
    result['crlb_theta'] = crlb_theta
    result['precision'] = precision
    result['success'] = result['nlopt_success'] and np.isfinite(crlb_theta)
    result['runtime'] = time.time() - start
    return result
