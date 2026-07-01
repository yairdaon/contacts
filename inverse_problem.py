import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.inverter import Inverter
from src.data_loader import load_synthetic
from src import flu, compute_g

OUTPUT_DIR = os.path.expanduser("~/contacts/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default population sizes for synthetic regions
DEFAULT_N = {
    "A": 1e7,
    "B": 1e6
}


def main(add_noise):

    disease = flu.Mortality
    theta = 0.05
    phase = np.zeros(2)

    regions = list(DEFAULT_N.keys())
    seasons = list(range(2010, 2019)) + [2023, 2024, 2025]


    # Build populations dictionary
    populations = {}
    for season in seasons:
        for region in regions:
            populations[(season, region)] = DEFAULT_N[region]

    

    noise_tag = "noisy" if add_noise else "clean"

    rows = []
    trajectories = {}

    # Similar ICs: both regions start near the same point
    n_seasons = len(seasons)
    np.random.seed(42)
    S_init = np.column_stack([np.random.uniform(*disease.slim_similar, size=n_seasons),
                              np.random.uniform(*disease.slim_similar, size=n_seasons)])
    I_init = np.column_stack([np.random.uniform(*disease.ilim_similar, size=n_seasons),
                              np.random.uniform(*disease.ilim_similar, size=n_seasons)])

    for phase2 in [0, np.pi]:
        phase[1] = phase2

        # Generate synthetic data
        obs, true_params = load_synthetic(
            disease=disease,
            regions=regions,
            seasons=seasons,
            theta=theta,
            phase=phase,
            populations=populations,
            S_init=S_init,
            I_init=I_init,
            add_noise=add_noise
        )

        ## Solve inverse problem
        inv = Inverter(
            phase=phase,
            obs=obs,
            disease=disease,
            populations=populations
        ).fit(n0=150)

        fitted = inv.objective.packer.unpack(inv.x)

        for i, season in enumerate(seasons):
            row = {
                'season': season,
                'objective': inv.fun,
                'success': inv.success,
                'status': inv.desc,
                'theta_true': theta,
                'theta_fit': fitted['theta'],
                'S1_0': fitted['S_init'][i, 0],
                'S2_0': fitted['S_init'][i, 1],
                'I1_0': fitted['I_init'][i, 0],
                'I2_0': fitted['I_init'][i, 1],
                'precision': inv.precisions[i],
                'phase2': phase2
            }
            rows.append(row)

        # Store first-season trajectories for plotting
        season = seasons[0]
        N = np.array([populations[(season, regions[0])],
                      populations[(season, regions[1])]])
        Ts = inv.objective.packer.all_Ts[season]

        # First-season observed data for plotting
        obs_season = obs[(obs['season'] == season)]

        trajectories[phase2] = {
            'true': compute_g.contacts(
                S0=true_params['S_init'][0, :] * N, I0=true_params['I_init'][0, :] * N,
                gamma=disease.gamma, theta=theta, Ts=Ts,
                beta0=disease.beta0, delta=disease.delta, phase=phase, N=N
            ).reset_index(),
            'fit': compute_g.contacts(
                S0=fitted['S_init'][0, :] * N, I0=fitted['I_init'][0, :] * N,
                gamma=disease.gamma, theta=fitted['theta'], Ts=Ts,
                beta0=disease.beta0, delta=disease.delta, phase=phase, N=N
            ).reset_index(),
            'obs': obs_season,
            'theta_fit': fitted['theta'],
        }

    df = pd.DataFrame(rows)

    # Compute aggregate std bound per phase (Fisher info adds under independence)
    crlb_std = {}
    for phase2 in [0, np.pi]:
        dd = df[df['phase2'] == phase2]
        crlb_std[phase2] = 1.0 / np.sqrt(np.sum(dd['precision']))

    # Plot: top = synchronized (phase2==0), bottom = unsynchronized (phase2==pi)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_idx, phase2 in enumerate([0, np.pi]):
        traj = trajectories[phase2]
        sync_label = "Synchronized" if phase2 == 0 else "Unsynchronized"
        std = crlb_std[phase2]

        for j, region in enumerate(regions):
            ax = axes[row_idx, j]
            true_j = traj['true'][traj['true']['j'] == j]
            fit_j = traj['fit'][traj['fit']['j'] == j]

            # S and I on left axis
            ax.plot(true_j['t'], true_j['S'], 'r-', label='S true')
            ax.plot(fit_j['t'], fit_j['S'], 'r--', label='S fitted')
            ax.plot(true_j['t'], true_j['I'], 'b-', label='I true')
            ax.plot(fit_j['t'], fit_j['I'], 'b--', label='I fitted')
            ax.set_ylabel('S, I')

            # Incidence on right axis (much smaller scale)
            ax2 = ax.twinx()
            ax2.plot(true_j['t'], true_j['mu'] * disease.rho, 'g-', alpha=0.7, label=r'$\rho\mu$ true')
            ax2.plot(fit_j['t'], fit_j['mu'] * disease.rho, 'g--', alpha=0.7, label=r'$\rho\mu$ fitted')
            obs_j = traj['obs'][traj['obs']['region'] == region].sort_values('t')
            ax2.plot(obs_j['t'], obs_j['incidence'], 'g.', markersize=10, alpha=0.5, label='Observed')
            ax2.set_ylabel('Incidence', color='g')
            ax2.tick_params(axis='y', labelcolor='g')

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')

            ax.set_title(f"{sync_label} - Region {region}\n"
                         rf"$\hat{{\theta}}$={traj['theta_fit']:.4f}, "
                         rf"$\sqrt{{\mathrm{{CRLB}}}}$={std:.4f}")
            ax.set_xlabel('t (years)')

    fig.suptitle(rf"$\theta_{{\mathrm{{true}}}}={theta}$", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"pix/inverse_problem_{noise_tag}.png", dpi=150, bbox_inches='tight')
    fig.savefig(f"pix/inverse_problem_{noise_tag}.pdf", bbox_inches='tight')
    plt.close()
    print(f"Synchronized {noise_tag}:   theta_fit={trajectories[0]['theta_fit']:.4f}, CRLB std={crlb_std[0]:.4f}")
    print(f"Unsynchronized {noise_tag}: theta_fit={trajectories[np.pi]['theta_fit']:.4f}, CRLB std={crlb_std[np.pi]:.4f}")


if __name__ == "__main__":
    try:
        main(add_noise=True)
        main(add_noise=False)
    except:
        import sys, traceback, pdb
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
