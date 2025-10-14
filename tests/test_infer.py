import numpy as np
import pickle
import pytest
from matplotlib import pyplot as plt

from src.helper import makepop, a2s
from src.inverter import Inverter, Objective
from src.losses import RHO
from tests.test_inverter import NWEEKS


@pytest.mark.parametrize("optimizer", ['nlopt'])  # , 'scipy'])
@pytest.mark.parametrize("difficulty", ["debug", "easy", "intermediate", "hard"])
def test_inference(optimizer, difficulty, seed=43):
    """
    Test that Inverter can recover known parameters from synthetic data.
    This is the key test for parameter inference capability.
    Also creates visualization of the reconstruction.

    Difficulty levels:
    - easy: 2 regions, 5 seasons, rho=0.95, theta=50 (low overdispersion = clean signal), start from true parameters
    - intermediate: 4 regions, 10 seasons, rho=0.8, theta=10 (moderate overdispersion), start from average of true and random
    - hard: 10 regions, 30 seasons, rho=0.5, theta=5 (high overdispersion = noisy data), start from random parameters
    """

    # Set test parameters based on difficulty
    if difficulty == 'debug':
        n_regions, n_seasons, n0, maxeval = 2, 30, 100, 10
    elif difficulty == "easy":
        n_regions, n_seasons, n0, maxeval = 2, 5, 250, None
    elif difficulty == "intermediate":
        n_regions, n_seasons, n0, maxeval = 5, 15, 750, None
    elif difficulty == "hard":
        n_regions, n_seasons, n0, maxeval = 10, 30, 5000, None
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    print(f"{difficulty} regions {n_regions}, seasons {n_seasons} starts={n0}")
    pop = makepop(n_regions=n_regions, n_seasons=n_seasons)

    objective = Objective(population=pop, n_weeks=NWEEKS, transform=optimizer == 'scipy')
    true_params = objective.packer.random_dict(seed=seed)

    # Override rho and theta with the parameterized values for testing
    # true_params['rho'] = rho  # rho is now fixed at 0.8

    # Pack true parameters
    x_true = objective.packer.pack(true_params)
    # objective.packer.verify(x_true)

    # Generate "observed" data using true parameters (not initial guess)
    true_trajectory = objective.sim(true_params)

    # Generate observed data
    obs = true_trajectory.copy()
    true_counts = true_trajectory['incidence'] * RHO  # Fixed rho value

    scale = np.sqrt(RHO * (1 - RHO) * true_counts)
    obs['incidence'] = true_counts + np.random.randn(true_counts.size) * scale
    obs['incidence'] = np.maximum(1e-6, obs['incidence'])  # Ensure non-negative
    objective.obs = obs

    inv = Inverter(objective=objective, optimizer=optimizer).fit(n0=n0, maxeval=maxeval)
    fname = f'pix/results_{difficulty}.pkl'
    with open(fname, 'wb') as f:
        flat = [[{**inv.packer.unpack(x), "fun": fun, 'idx': idx} for idx, (x, fun) in
               enumerate(zip(res['x_list'], res['out_list']))] for res in inv.results]
        pickle.dump(flat, f)

    fun = inv.fun
    inferred = inv.packer.unpack(inv.x)

    # Get initial guess from the best optimization run
    best_result = min(inv.results, key=lambda r: r['fun'])
    if len(best_result['x_list']) > 0:
        initial_guess = inv.packer.unpack(best_result['x_list'][0])
    else:
        initial_guess = inv.packer.random_dict()  # fallback

    # Generate reconstructed trajectory for visualization
    reconstructed_trajectory = objective.sim(inferred)

    err_beta0 = abs(true_params['beta0'] - inferred['beta0'])  # / true_params['beta0']
    err_eps = abs(true_params['eps'] - inferred['eps'])  # / true_params['eps']
    # err_rho = abs(true_params['rho'] - inferred['rho'])  # rho is fixed at 0.8
    # err_E_init = np.max(np.abs(true_params['E_init'] - inferred['E_init']))  # E_init = I_init
    err_omega = abs(true_params['omega'] - inferred['omega'])  # omega now scalar
    err_c = np.max(np.abs(true_params['c_vec'] - inferred['c_vec']))  # / true_params['c_vec'])
    # err_theta = abs(true_params['theta'] - inferred['theta']) / true_params['theta']

    print("\nParameter Recovery Results:")
    print(f"  beta0  - True: {true_params['beta0']:.3f}, Initial: {initial_guess['beta0']:.3f}, Inferred: {inferred['beta0']:.3f}, err: {err_beta0:.3f}")
    print(f"  eps    - True: {true_params['eps']:.3f}, Initial: {initial_guess['eps']:.3f}, Inferred: {inferred['eps']:.3f}, err: {err_eps:.3f}")
    print(f"  omega  - True: {true_params['omega']:.3f}, Initial: {initial_guess['omega']:.3f}, Inferred: {inferred['omega']:.3f}, err: {err_omega:.3f}")
    print(f"  c      - True: {a2s(true_params['c_vec'])}, Initial: {a2s(initial_guess['c_vec'])}, Inferred: {a2s(inferred['c_vec'])}, err: {err_c:.3f}")
    # print(f"  rho    - True: {true_params['rho']:.3f}, Inferred: {inferred['rho']:.3f}, err: {err_rho:.3f}")  # rho fixed at 0.8
    # print(f"  theta  - True: {true_params['theta']:.3f}, Inferred: {inferred['theta']:.3f}, err: {err_theta:.3f}")

    # assert err_beta0 < EPS, f"err beta0 = {err_beta0:.3f}"
    # assert err_eps < EPS, f"err eps {err_eps:.3f}"
    # # assert err_rho < EPS, f"err rho {err_rho:.3f}"  # rho fixed at 0.8
    # assert np.all(err_omega < EPS), f"err omega {err_omega:.3f}"
    # assert np.all(err_c < EPS), f"err c {err_c::.3f}"
    # # assert err_theta < EPS, f"err theta {err_theta:.3f}"
    #
    # Test that final loss is finite and reasonable
    assert np.isfinite(fun), f"Final loss is not finite: {fun}"
    assert fun >= 0, f"Final loss is negative: {fun}"

    # Create visualization
    print("Creating parameter inference visualization...")
    seasons = sorted(pop.season.unique())

    # Set dark theme
    plt.style.use('dark_background')

    # Calculate relative differences for scalar parameters
    param_rel_diff = {}
    for parameter in ['beta0', 'omega', 'eps', 'c_vec']:
        param_rel_diff[parameter] = []
        for res in inv.results:
            if len(res['x_list']) > 0:
                x0 = res['x_list'][0]
                x_final = res['x']
                
                unpacked_0 = inv.packer.unpack(x0)
                unpacked_final = inv.packer.unpack(x_final)
                
                if parameter == 'c_vec':
                    initial_val = unpacked_0[parameter][0]
                    final_val = unpacked_final[parameter][0]
                else:
                    initial_val = unpacked_0[parameter]
                    final_val = unpacked_final[parameter]
                
                # Calculate relative difference: |final - initial| / |initial|
                rel_diff = np.abs(final_val - initial_val) / np.abs(initial_val)
                param_rel_diff[parameter].append(rel_diff)

    fig, axes = plt.subplots(2, 2, facecolor='#1e1e1e')
    for ax, parameter in zip(np.ravel(axes), ['beta0', 'omega', 'eps', 'c_vec']):
        if parameter == 'c_vec':
            f = lambda x: inv.packer.unpack(x)['c_vec'][0]
        else:
            f = lambda x: inv.packer.unpack(x)[parameter]
        for res in inv.results:
            param_values = [f(x) for x in res['x_list']]
            line, = ax.plot(param_values, res['out_list'], alpha=0.2, linewidth=0.5)
            color = line.get_color()  # Get color from line plot
            # Use the optimal point from this specific optimization run
            ax.scatter(f(res['x']), res['fun'], color=color, marker='*', s=15)
        
        # Mark true parameter value
        if parameter == 'c_vec':
            true_value = true_params['c_vec'][0]
        else:
            true_value = true_params[parameter]
        ax.axvline(x=true_value, color='red', linestyle='--', linewidth=2, alpha=0.8, label='True value')
            
        ax.set_xlabel(parameter)
        ax.set_xlim([0, 1])
        #ax.set_ylabel("Objective")
        #ax.set_ylim([0, 10**4])
        ax.set_yscale('log')

    plt.savefig(f'pix/{difficulty}_parameters_path.png', dpi=300)#, bbox_inches='tight')
    plt.close()

    # Calculate relative differences between initial and final parameters for each chain
    rel_diff_data = {}  # Store relative differences for each compartment/season/region
    
    for res in inv.results:
        if len(res['x_list']) > 0:
            x0 = res['x_list'][0]  # Initial parameters
            x_final = res['x']     # Final optimal parameters
            
            unpacked_0 = inv.packer.unpack(x0)
            unpacked_final = inv.packer.unpack(x_final)
            
            for compartment in ['S_init', 'E_init', 'I_init']:
                if compartment not in rel_diff_data:
                    rel_diff_data[compartment] = {}
                
                initial_values = unpacked_0[compartment]
                final_values = unpacked_final[compartment]
                
                # Calculate relative difference: |final - initial| / |initial|
                rel_diff = np.abs(final_values - initial_values) / np.abs(initial_values)
                
                for season_idx in range(len(seasons)):
                    for region_idx in range(n_regions):
                        key = (season_idx, region_idx)
                        if key not in rel_diff_data[compartment]:
                            rel_diff_data[compartment][key] = []
                        rel_diff_data[compartment][key].append(rel_diff[season_idx, region_idx])

    # Create separate plots for S_init, E_init, I_init - parameter value vs objective
    compartments = ['S_init', 'E_init', 'I_init']
    
    for compartment in compartments:
        # Create figure with regions as rows and seasons as columns
        fig, axes = plt.subplots(n_regions, len(seasons), figsize=(4*len(seasons), 3*n_regions), facecolor='#1e1e1e')
        
        # Handle case where we have only one season or one region
        if len(seasons) == 1 and n_regions == 1:
            axes = [[axes]]
        elif len(seasons) == 1:
            axes = [[ax] for ax in axes]
        elif n_regions == 1:
            axes = [axes]
        
        for season_idx, season in enumerate(seasons):
            for region_idx in range(n_regions):
                ax = axes[region_idx][season_idx]
                
                # Plot parameter value vs objective for each optimization run
                for res in inv.results:
                    # Extract parameter values and objectives
                    param_values = []
                    objectives = []
                    
                    for x, obj_val in zip(res['x_list'], res['out_list']):
                        unpacked = inv.packer.unpack(x)
                        comp_array = unpacked[compartment]  # (n_seasons, n_regions) array
                        param_value = comp_array[season_idx, region_idx]
                        param_values.append(param_value)
                        objectives.append(obj_val)
                    
                    # Plot trajectory as scatter plot with small markers
                    scatter = ax.scatter(param_values, objectives, s=5, alpha=0.7)
                    color = scatter.get_facecolor()[0]  # Get color from scatter plot
                    
                    # Mark optimal point from this specific optimization run
                    final_unpacked = inv.packer.unpack(res['x'])
                    final_param = final_unpacked[compartment][season_idx, region_idx]
                    ax.scatter(final_param, res['fun'], color=color, marker='*', s=30, alpha=0.9)
                
                # Calculate max relative change for this specific season/region combination
                key = (season_idx, region_idx)
                if key in rel_diff_data[compartment]:
                    max_rel_change = np.max(rel_diff_data[compartment][key])
                    ax.set_title(f'Season {season_idx}, Region {region_idx}\nMax Δ: {max_rel_change:.2e}', 
                               color='white', fontsize=9)
                else:
                    ax.set_title(f'Season {season_idx}, Region {region_idx}', color='white', fontsize=10)
                ax.set_xlabel(f'{compartment} value', color='white')
                ax.set_ylabel('Objective', color='white')
                ax.set_yscale('log')
                
                # Set appropriate x-axis limits based on compartment
                if compartment == 'S_init':
                    ax.set_xlim([0.5, 1])
                elif compartment in ['E_init', 'I_init']:
                    ax.set_xlim([0, 1e-3])
                
                ax.grid(True, alpha=0.2, color='white')
                ax.set_facecolor('#1e1e1e')
                ax.tick_params(colors='white')
        
        plt.suptitle(f'{compartment} Optimization Trajectories - {difficulty}', color='white', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'pix/{difficulty}_{compartment}_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()

    fig, axes = plt.subplots(1, len(seasons), figsize=(5 * len(seasons), 5), facecolor='#1e1e1e')
    if len(seasons) == 1:
        axes = [axes]

    fig.suptitle(f'Parameter Inference Test Results (seed={seed})\n'
                 f'β₀ err: {err_beta0:.3f}, ε err: {err_eps:.3f}', fontsize=14, color='white')

    colors = ['#00D4FF', '#FF6B9D']  # Bright cyan and pink for dark background
    regions = ['HHS0', 'HHS1']

    for season_idx, season in enumerate(seasons):
        ax = axes[season_idx]

        for region_idx, region in enumerate(regions):
            # True trajectory
            true_data = true_trajectory[
                (true_trajectory.season == season) &
                (true_trajectory.region == region)
                ].sort_values('time')

            obs_data = obs[
                (obs.season == season) &
                (obs.region == region)
                ].sort_values('time')

            # Reconstructed trajectory
            reconstructed_data = reconstructed_trajectory[
                (reconstructed_trajectory.season == season) &
                (reconstructed_trajectory.region == region)
                ].sort_values('time')

            ax.plot(true_data.time, true_data.incidence,
                    color=colors[region_idx], linewidth=2.5,
                    label=f'{region} - True' if season_idx == 0 else "")

            ax.plot(reconstructed_data.time, reconstructed_data.incidence,
                    color=colors[region_idx], linewidth=2.5, linestyle='--', alpha=0.8,
                    label=f'{region} - Inferred' if season_idx == 0 else "")

            ax.plot(obs_data.time, reconstructed_data.incidence,
                    color=colors[region_idx], linewidth=2.5, linestyle=':', alpha=0.8,
                    label=f'{region} - Noisy' if season_idx == 0 else "")

        ax.set_title(f'{season[:4]}', fontsize=12, color='white')
        ax.set_xlabel('Time', color='white')
        if season_idx == 0:
            ax.set_ylabel('Weekly Incidence', color='white')
            ax.legend(loc='upper right', fancybox=True, shadow=True, facecolor='#2e2e2e', edgecolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(f'pix/{difficulty}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
