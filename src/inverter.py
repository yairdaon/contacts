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
                 populations):

        self.objective = Objective(obs=obs,
                                   phase=phase,
                                   disease=disease,
                                   populations=populations)

    def fit(self,
            n0,
            n_jobs=-1,
            fname=None):

        results = Parallel(n_jobs=n_jobs)(delayed(single_optimization)(self.objective) for _ in tqdm(range(n0)))

        n_ok = sum(1 for r in results if r['success'])
        n_fail = n0 - n_ok
        print(f"Optimization: {n_ok} succeeded + {n_fail} failed / {n0} total")

        best = min(results, key=lambda r: r['fun'])
        self.x = best['x']
        self.success = best['success']
        self.fun = best['fun']
        self.nlopt_code = best['nlopt_code']
        self.desc = best['desc']
        self.runtime = best['runtime']
        self.precisions = best['precisions']

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
    opt.set_upper_bounds([1.0] * (2 * M) + [0.5])

    x0 = packer.random_vector()
    try:
        x = opt.optimize(x0)
        code = opt.last_optimize_result()
        result = dict(x=x, fun=opt.last_optimum_value(), nlopt_success=1<=code<=4, nlopt_code=code, desc=CODES[code])
    except Exception:
        fun_x0 = objective(x0)
        result = dict(x=x0, fun=fun_x0, nlopt_success=False, nlopt_code=-1, desc="failed")

    failed_seasons = []
    # Compute Fisher information for each season
    fitted = packer.unpack(result['x'])
    precisions = []
    for season_idx, season in enumerate(packer.seasons):
        N = np.array([packer.populations[(season, packer.regions[i])]
                      for i in range(n_regions)])
        try:
            precision = compute_precision(
                S0=fitted['S_init'][season_idx, :] * N,
                I0=fitted['I_init'][season_idx, :] * N,
                gamma=objective.disease.gamma,
                theta=fitted['theta'],
                Ts=packer.all_Ts[season],
                beta0=objective.disease.beta0,
                delta=objective.disease.delta,
                rho=objective.disease.rho,
                phase=objective.phase,
                N=N
            )
            precisions.append(precision)
        except Exception:
            failed_seasons.append(str(season))
            precisions.append(np.nan)

    if failed_seasons:
        result['desc'] += f" | failed precisions: {', '.join(failed_seasons)}"
    result['precisions'] = precisions
    result['success'] = result['nlopt_success'] and all(np.isfinite(c) for c in precisions)
    result['runtime'] = time.time() - start
    return result
