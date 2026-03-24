import os
import numpy as np
import pandas as pd
import plac
import nlopt
import matplotlib.pyplot as plt

from src.inverter import Inverter
from src.data_loader import load_synthetic
from src import flu, compute_g
from src.crlb import compute_crlb

OUTPUT_DIR = os.path.expanduser("~/contacts/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default population sizes for synthetic regions
DEFAULT_N = {
    "California": 1e6,
    "Alaska": 1e6
}


def main():

    disease = flu.ILI
    disease = flu.Mortality
    theta = 0.05
    phase = np.zeros(2)

    regions = list(DEFAULT_N.keys())
    seasons = list(range(1995, 2005))  # 10 seasons

    # Build populations dictionary
    populations = {}
    for season in seasons:
        for region in regions:
            populations[(season, region)] = DEFAULT_N[region]

    n0 = 25
    maxeval = None
    optimizer = nlopt.LD_SLSQP
    # optimizer = nlopt.LD_MMA
    # optimizer = nlopt.LD_CCSAQ

    rows = []
    trajectories = {}

    for phase2 in [0, np.pi]:
        phase[1] = phase2

        # Generate synthetic data
        obs, true_params = load_synthetic(
            disease=disease,
            regions=regions,
            seasons=seasons,
            theta=theta,
            phase=phase,
            populations=populations
        )

        ## Solve inverse problem
        inv = Inverter(optimizer=optimizer,
                       phase=phase,
                       obs=obs,
                       disease=disease,
                       populations=populations).fit(n0=n0, maxeval=maxeval)

        fitted = inv.objective.packer.unpack(inv.x)

        for i, season in enumerate(seasons):
            N = np.array([populations[(season, regions[0])],
                          populations[(season, regions[1])]])

            try:
                bound = compute_crlb(
                    S0=fitted['S_init'][i, :],
                    I0=fitted['I_init'][i, :],
                    gamma=disease.gamma,
                    theta=fitted['theta'],
                    Ts=inv.objective.packer.all_Ts[season],
                    beta0=disease.beta0,
                    eps=disease.eps,
                    rho=disease.rho,
                    phase=phase,
                    N=N
                )
                err = ''
            except Exception as e:
                bound = np.nan
                err = str(e)

            row = {
                'season': season,
                'objective': inv.fun,
                'success': inv.success,
                'nlopt_code': inv.nlopt_code,
                'theta_true': theta,
                'theta_fit': fitted['theta'],
                'S1_0': fitted['S_init'][i, 0],
                'S2_0': fitted['S_init'][i, 1],
                'I1_0': fitted['I_init'][i, 0],
                'I2_0': fitted['I_init'][i, 1],
                'crlb': bound,
                'error': err,
                'phase2': phase2
            }
            rows.append(row)

        # Store first-season trajectories for plotting
        season = seasons[0]
        N = np.array([populations[(season, regions[0])],
                      populations[(season, regions[1])]])
        Ts = inv.objective.packer.all_Ts[season]

        trajectories[phase2] = {
            'true': compute_g.contacts(
                S0=true_params['S_init'][0, :], I0=true_params['I_init'][0, :],
                gamma=disease.gamma, theta=theta, Ts=Ts,
                beta0=disease.beta0, eps=disease.eps, phase=phase, N=N
            ).reset_index(),
            'fit': compute_g.contacts(
                S0=fitted['S_init'][0, :], I0=fitted['I_init'][0, :],
                gamma=disease.gamma, theta=fitted['theta'], Ts=Ts,
                beta0=disease.beta0, eps=disease.eps, phase=phase, N=N
            ).reset_index(),
            'theta_fit': fitted['theta'],
        }

    df = pd.DataFrame(rows)

    # Plot: top = synchronized (phase2==0), bottom = unsynchronized (phase2==pi)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_idx, phase2 in enumerate([0, np.pi]):
        traj = trajectories[phase2]
        sync_label = "Synchronized" if phase2 == 0 else "Unsynchronized"

        for j, region in enumerate(regions):
            ax = axes[row_idx, j]
            true_j = traj['true'][traj['true']['j'] == j]
            fit_j = traj['fit'][traj['fit']['j'] == j]

            ax.plot(true_j['t'], true_j['S'], 'r-', label='S true')
            ax.plot(fit_j['t'], fit_j['S'], 'r--', label='S fitted')
            ax.plot(true_j['t'], true_j['I'], 'b-', label='I true')
            ax.plot(fit_j['t'], fit_j['I'], 'b--', label='I fitted')

            ax.set_title(f"{sync_label} - {region} (theta_fit={traj['theta_fit']:.4f})")
            ax.set_xlabel('t')
            ax.legend()

    fig.suptitle(f"theta_true={theta}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
