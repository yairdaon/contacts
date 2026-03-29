import nlopt
import numpy as np
import time
from joblib import delayed, Parallel
from tqdm import tqdm

from src.objective import Objective
from src.crlb import compute_crlb
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
            plot=False):

        results = Parallel(n_jobs=n_jobs)(delayed(single_optimization)(self.objective) for _ in tqdm(range(n0)))
        best = min(results, key=lambda r: r['fun'])
        self.x = best['x']
        self.success = best['success']
        self.fun = best['fun']
        self.nlopt_code = best['nlopt_code']
        self.desc = best['desc']
        self.runtime = best['runtime']
        self.crlbs = best['crlbs']

        if plot:
            self._plot_reconstruction()

        return self

    def _plot_reconstruction(self):
        """Plot observed vs fitted incidence for each region and season."""
        import matplotlib.pyplot as plt

        obj = self.objective
        packer = obj.packer
        fitted = packer.unpack(self.x)

        sim = packer.sim(fitted, obj.phase, obj.disease)
        obs = obj.obs

        regions = list(packer.regions)
        seasons = list(packer.seasons)[:2]
        n_seasons = len(seasons)

        fig, axes = plt.subplots(n_seasons, len(regions), figsize=(5 * len(regions), 3 * n_seasons),
                                 squeeze=False)

        for i, season in enumerate(seasons):
            for j, region in enumerate(regions):
                ax = axes[i, j]
                obs_r = obs[(obs['season'] == season) & (obs['region'] == region)].sort_values('t')
                sim_r = sim[(sim['season'] == season) & (sim['region'] == region)].sort_values('t')

                ax.plot(obs_r['t'], obs_r['incidence'], 'ko', markersize=3, label='Observed')
                ax.plot(sim_r['t'], sim_r['mu'] * obj.disease.rho, 'r-', linewidth=1.5, label=r'$\rho\mu$')

                if i == 0:
                    ax.set_title(region)
                if j == 0:
                    ax.set_ylabel(f'{season}')
                if i == n_seasons - 1:
                    ax.set_xlabel('t')
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)

        fig.suptitle(rf"$\hat{{\theta}}={fitted['theta']:.4f}$, status={self.desc}", fontsize=12)
        fig.tight_layout()
        plt.show()


def single_optimization(objective):

    start = time.time()
    packer = objective.packer
    n_regions = packer.n_regions
    n_seasons = packer.n_seasons
    M = n_regions * n_seasons
    n = packer.n_params

    def make_opt():
        opt = nlopt.opt(nlopt.LD_SLSQP, n)
        opt.set_ftol_rel(1e-8)
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
        opt.set_upper_bounds([1.0] * (2 * M) + [0.1])
        return opt

    # Stage 1: unweighted MSE (robust to bad starting points)
    objective.weighted = False
    opt = make_opt()
    x0 = packer.random_vector()
    try:
        x0 = opt.optimize(x0)
    except Exception:
        pass  # use whatever x0 we have

    # Stage 2: weighted NLL (statistically efficient, starting from MSE solution)
    objective.weighted = True
    opt = make_opt()
    try:
        x = opt.optimize(x0)
        code = opt.last_optimize_result()
        result = dict(x=x, fun=opt.last_optimum_value(), nlopt_success=1<=code<=4, nlopt_code=code, desc=CODES[code])
    except Exception:
        # Weighted stage failed — fall back to MSE solution
        fun_x0 = objective(x0)
        result = dict(x=x0, fun=fun_x0, nlopt_success=True, nlopt_code=3, desc="FTOL_REACHED (MSE only)")

    failed_seasons = []
    # Compute CRLBs for each season
    fitted = packer.unpack(result['x'])
    crlbs = []
    for season_idx, season in enumerate(packer.seasons):
        N = np.array([packer.populations[(season, packer.regions[i])]
                      for i in range(n_regions)])
        try:
            bound = compute_crlb(
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
            crlbs.append(bound)
        except Exception:
            failed_seasons.append(str(season))
            crlbs.append(np.nan)

    if failed_seasons:
        result['desc'] += f" | failed crlbs: {', '.join(failed_seasons)}"
    result['crlbs'] = crlbs
    result['success'] = result['nlopt_success'] and all(np.isfinite(c) for c in crlbs)
    result['runtime'] = time.time() - start
    return result
