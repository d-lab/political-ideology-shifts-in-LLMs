import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import distance

warnings.filterwarnings('ignore')

def setup_paths():
    """Ensures the project root is in the system path."""
    try:
        # Assumes the script is in a 'scripts' or similar subfolder
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        if proj_root not in sys.path:
            sys.path.append(proj_root)
        print(f"Project root set to: {proj_root}")
    except NameError:
        # Fallback for interactive environments
        proj_root = os.path.abspath(os.path.join(os.pardir))
        if proj_root not in sys.path:
            sys.path.append(proj_root)
        print(f"Project root set to (fallback): {proj_root}")
    return proj_root

PROJ_ROOT = setup_paths()


def load_compass_data(model: str, setting: str, use_template: bool) -> pd.DataFrame:
    if use_template:
        file_path = os.path.join(PROJ_ROOT, 'results', model, setting, 'final_compass.pqt')
    else:
        file_path = os.path.join(PROJ_ROOT, 'results', model, setting, 'final_compass_no_chat_template.pqt')
    
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found at: {file_path}. Skipping this configuration.")
        return None  # Return None if file is not found

    df = pd.read_parquet(file_path)
    df['compass_position'] = df['compass_position'].apply(lambda x: (x[0], x[1]))
    unique_df = df[['persona_id', 'compass_position']].drop_duplicates().reset_index(drop=True)
    return unique_df

def load_no_persona_compass_data(model: str, use_template: bool) -> tuple:
    """Load the no-persona LLM leaning data and return compass position."""
    if use_template:
        file_path = os.path.join(PROJ_ROOT, 'results', model, 'no_persona_LLM_leaning.pqt')
        print(file_path)
    else:
        file_path = os.path.join(PROJ_ROOT, 'results', model, 'no_persona_LLM_leaning_no_chat_template.pqt')
    
    if not os.path.exists(file_path):
        print(f"Warning: No-persona data file not found at: {file_path}")
        return None
    
    df = pd.read_parquet(file_path)
    
    # Get compass position from any row (they should all be the same)
    compass_position = df['compass_position'].iloc[0]

    # Handle different formats
    if isinstance(compass_position, (list, tuple)):
        return tuple(compass_position)
    elif isinstance(compass_position, dict):
        return (compass_position.get('economic', 0), compass_position.get('social', 0))
    else:
        print(f"Warning: Unexpected compass_position format: {type(compass_position)}")
        return None


def calculate_centroid(df: pd.DataFrame) -> np.ndarray:
    positions = np.array([np.array(pos) for pos in df['compass_position']])
    return np.mean(positions, axis=0)


def plot_compass_density(df, bins=None, gridsize=30, label_gap=1.5, label_margin=0.15,
                         marginal_ratio=12, tick_interval=2, scientific_notation=True,
                         label_size=10, tick_size=8, colorbar_size=8, marginal_alpha=0.5,
                         origin_lines_alpha=0.8, np_dot=None, triangle=False, color='green', circle_dot=None):

    if color == 'green':
        distribution_color = "Greens"
        marginals_color = '#228B22'  # Default green color
    elif color == 'blue':
        distribution_color = "Blues"
        marginals_color = '#1E90FF'
    elif color == 'yellow':
        distribution_color = "YlOrBr"
        marginals_color = '#FFD700'  # Gold color for yellow
    elif color == 'red':
        # distribution_color = "YlOrRd"
        distribution_color = "Reds"
        marginals_color = '#FF6347'
    else:
        raise ValueError("Invalid color choice. Use 'green', 'blue', or 'red'.")

    x_coords = [pos[0] for pos in df['compass_position']]
    y_coords = [pos[1] for pos in df['compass_position']]

    g = sns.JointGrid(x=x_coords, y=y_coords, height=8, ratio=marginal_ratio, marginal_ticks=True, space=label_margin)

    hexbin = g.ax_joint.hexbin(
        x_coords, y_coords, gridsize=gridsize, cmap=distribution_color,
        mincnt=1, bins=bins, extent=(-10, 10, -10, 10), edgecolor='#DCDCDC', linewidth=0.5
    )

    g.plot_marginals(sns.histplot, bins=gridsize, color=marginals_color, alpha=marginal_alpha)
    g.figure.set_size_inches(10, 10)
    g.ax_joint.set_xlim(-10, 10)
    g.ax_joint.set_ylim(-10, 10)

    ticks = np.arange(-10, 11, tick_interval)
    g.ax_joint.set_xticks(ticks)
    g.ax_joint.set_yticks(ticks)

    grid_lines = np.arange(-10, 11, 1)
    g.ax_joint.set_xticks(grid_lines, minor=True)
    g.ax_joint.set_yticks(grid_lines, minor=True)

    for spine in ['top', 'right', 'left', 'bottom']:
        g.ax_joint.spines[spine].set_visible(True)
        g.ax_joint.spines[spine].set_color('black')
        g.ax_joint.spines[spine].set_linewidth(1.2)

    def custom_formatter(x, p):
        if x == -10: return ''
        if abs(x) >= 1000: return f'{int(x/1000)}k'
        return str(int(x))

    g.ax_joint.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    g.ax_joint.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

    def format_thousand_ticks(x, p):
        if abs(x) >= 1000: return f'{int(x/1000)}k'
        return str(int(x))

    g.ax_marg_x.yaxis.set_major_formatter(plt.FuncFormatter(format_thousand_ticks))
    g.ax_marg_y.xaxis.set_major_formatter(plt.FuncFormatter(format_thousand_ticks))

    g.ax_joint.tick_params(axis='both', labelsize=tick_size)
    g.ax_marg_x.tick_params(axis='both', labelsize=tick_size)
    g.ax_marg_y.tick_params(axis='both', labelsize=tick_size)

    g.ax_joint.set_xlabel('')
    g.ax_joint.set_ylabel('')

    plt.subplots_adjust(left=0.15 + (label_gap / 20), bottom=0.15 + (label_gap / 20))

    label_pos = 10 + label_gap
    g.ax_joint.text(-10, -label_pos, '← Left', ha='left', va='top', transform=g.ax_joint.transData, fontsize=label_size)
    g.ax_joint.text(10, -label_pos, 'Right →', ha='right', va='top', transform=g.ax_joint.transData, fontsize=label_size)
    g.ax_joint.text(-label_pos, 10, 'Authorit. →', ha='right', va='top', rotation=90, transform=g.ax_joint.transData, fontsize=label_size)
    g.ax_joint.text(-label_pos, -10, '← Libert.', ha='right', va='bottom', rotation=90, transform=g.ax_joint.transData, fontsize=label_size)
    g.ax_joint.text(-10.3, -10.3, '-10', ha='right', va='top', transform=g.ax_joint.transData, fontsize=tick_size)

    joint_pos = g.ax_joint.get_position()
    marg_right_pos = g.ax_marg_y.get_position()
    cax = g.figure.add_axes([marg_right_pos.x1 + 0.04, joint_pos.y0, 0.03, joint_pos.height])

    cbar = g.figure.colorbar(hexbin, cax=cax, label='')
    if not scientific_notation:
        cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.tick_params(labelsize=colorbar_size)
    cbar.update_ticks()

    g.ax_joint.grid(True, which='major', linestyle='-', alpha=0.7)
    g.ax_joint.grid(True, which='minor', linestyle='--', alpha=0.7)
    g.ax_joint.axhline(y=0, color='k', linestyle='-', alpha=origin_lines_alpha, linewidth=1.5)
    g.ax_joint.axvline(x=0, color='k', linestyle='-', alpha=origin_lines_alpha, linewidth=1.5)

    if np_dot is not None:
        if triangle:
            g.ax_joint.scatter(np_dot[0], np_dot[1], marker='^', color='black', s=500, zorder=4)
            g.ax_joint.scatter(np_dot[0], np_dot[1], marker='^', color='white', s=358, zorder=5)
        else:
            g.ax_joint.scatter(np_dot[0], np_dot[1], color='black', s=150, zorder=4)
            g.ax_joint.scatter(np_dot[0], np_dot[1], color='white', s=100, zorder=5)

    # Plot circle marker (for base setting with no-persona data)
    if circle_dot is not None:
        g.ax_joint.scatter(circle_dot[0], circle_dot[1], marker='o', color='black', s=500, zorder=4)
        g.ax_joint.scatter(circle_dot[0], circle_dot[1], marker='o', color='white', s=358, zorder=5)

    return g


def run_centroid_analysis(df: pd.DataFrame):
    # Extract coordinates
    coords = np.array(df['compass_position'].tolist())
    
    # Calculate centroid
    centroid = np.mean(coords, axis=0)
    
    # Calculate Euclidean distance of each point from the centroid
    distances = [distance.euclidean(point, centroid) for point in coords]
    
    # Return the average distance
    return np.mean(distances)


def run_significance_test(df_base: pd.DataFrame, df_ideology: pd.DataFrame):
    df_merged = df_base.merge(df_ideology, on='persona_id', suffixes=('_base', '_ideology'))

    # Extract x and y coordinates
    x_base = np.array([pos[0] for pos in df_merged['compass_position_base']])
    x_ideology = np.array([pos[0] for pos in df_merged['compass_position_ideology']])
    y_base = np.array([pos[1] for pos in df_merged['compass_position_base']])
    y_ideology = np.array([pos[1] for pos in df_merged['compass_position_ideology']])

    # Calculate differences
    x_diff = x_ideology - x_base
    y_diff = y_ideology - y_base
    n = len(x_diff)  # sample size

    # Basic statistics
    print()
    print("="*70)
    print("--- Basic Statistics ---")
    print(f"\nX-axis mean difference: {np.mean(x_diff):.3f}")
    print(f"X-axis std difference: {np.std(x_diff):.3f}")
    print(f"Y-axis mean difference: {np.mean(y_diff):.3f}")
    print(f"Y-axis std difference: {np.std(y_diff):.3f}")
    print("="*70)
    print()

    # 1. Check normality of differences
    _, p_norm_x = stats.normaltest(x_diff)
    _, p_norm_y = stats.normaltest(y_diff)

    print()
    print("="*70)
    print("Normality test p-values (D'Agostino-Pearson test):")
    print(f"\nX-axis differences: p = {p_norm_x:.6f}")
    print(f"Y-axis differences: p = {p_norm_y:.6f}")
    print("="*70)
    print()

    # Wilcoxon signed-rank test with z-score calculation
    w_stat_x, w_p_val_x = stats.wilcoxon(x_base, x_ideology)
    w_stat_y, w_p_val_y = stats.wilcoxon(y_base, y_ideology)

    # Calculate z-scores for Wilcoxon test
    # z = (W - n(n+1)/4) / sqrt(n(n+1)(2n+1)/24)
    def wilcoxon_z(w_stat, n):
        expected_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        return (w_stat - expected_w) / std_w

    z_score_x = wilcoxon_z(w_stat_x, n)
    z_score_y = wilcoxon_z(w_stat_y, n)

    print()
    print("="*70)
    print("Wilcoxon signed-rank test results:")
    print(f"\nX-axis: statistic = {w_stat_x:.3f}, z-score = {z_score_x:.3f}, p-value = {w_p_val_x:.6f}")
    print(f"Y-axis: statistic = {w_stat_y:.3f}, z-score = {z_score_y:.3f}, p-value = {w_p_val_y:.6f}")
    print("="*70)
    print()

    # Effect sizes (Cohen's d) with confidence intervals and p-values
    def cohens_d_with_stats(x1, x2):
        n = len(x1)
        d = (np.mean(x1) - np.mean(x2)) / np.sqrt(
            ((n - 1) * np.var(x1, ddof=1) + (n - 1) * np.var(x2, ddof=1)) / (2 * n - 2)
        )
        
        # Standard error of d
        se = np.sqrt((4/n) + (d*d)/(2*n))  # approximation for paired samples
        
        # Calculate 95% confidence interval
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se
        
        # Calculate z-score and p-value for Cohen's d
        z_score = d / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # two-tailed test
        
        return d, ci_lower, ci_upper, z_score, p_value

    d_x, ci_lower_x, ci_upper_x, z_d_x, p_d_x = cohens_d_with_stats(x_ideology, x_base)
    d_y, ci_lower_y, ci_upper_y, z_d_y, p_d_y = cohens_d_with_stats(y_ideology, y_base)

    print()
    print("="*70)
    print("Effect sizes (Cohen's d):")
    print(f"\nX-axis: d = {d_x:.3f}, 95% CI [{ci_lower_x:.3f}, {ci_upper_x:.3f}], z = {z_d_x:.3f}, p = {p_d_x:.6f}")
    print(f"Y-axis: d = {d_y:.3f}, 95% CI [{ci_lower_y:.3f}, {ci_upper_y:.3f}], z = {z_d_y:.3f}, p = {p_d_y:.6f}")
    print("="*70)
    print()


def run_coverage_analysis(df: pd.DataFrame, gridsize: int, setting_name: str):
    print()
    print('-' * 70)
    print(f"--- Coverage Analysis for '{setting_name}' setting ---")

    x_coords = [pos[0] for pos in df['compass_position']]
    y_coords = [pos[1] for pos in df['compass_position']]
    extent = [-10, 10, -10, 10]

    # --- Step 1: Calculate the total number of possible hexbins in the entire plot and in each quadrant ---
    x_dummy = np.linspace(extent[0], extent[1], 800)
    y_dummy = np.linspace(extent[2], extent[3], 800)
    xx, yy = np.meshgrid(x_dummy, y_dummy)

    fig, ax = plt.subplots()
    all_bins_collection = ax.hexbin(xx.flatten(), yy.flatten(), gridsize=gridsize, extent=extent)
    plt.close(fig)

    all_bin_centers = all_bins_collection.get_offsets()
    total_possible_bins = len(all_bin_centers)

    total_q_tr = np.sum((all_bin_centers[:, 0] > 0) & (all_bin_centers[:, 1] > 0))
    total_q_tl = np.sum((all_bin_centers[:, 0] < 0) & (all_bin_centers[:, 1] > 0))
    total_q_bl = np.sum((all_bin_centers[:, 0] < 0) & (all_bin_centers[:, 1] < 0))
    total_q_br = np.sum((all_bin_centers[:, 0] > 0) & (all_bin_centers[:, 1] < 0))

    # --- Step 2: Calculate the number of active hexbins from the actual data ---
    fig, ax = plt.subplots()
    active_bins_collection = ax.hexbin(x_coords, y_coords, gridsize=gridsize, extent=extent, mincnt=1)
    plt.close(fig)

    active_bin_centers = active_bins_collection.get_offsets()
    total_active_bins = len(active_bin_centers)

    active_q_tr = np.sum((active_bin_centers[:, 0] > 0) & (active_bin_centers[:, 1] > 0))
    active_q_tl = np.sum((active_bin_centers[:, 0] < 0) & (active_bin_centers[:, 1] > 0))
    active_q_bl = np.sum((active_bin_centers[:, 0] < 0) & (active_bin_centers[:, 1] < 0))
    active_q_br = np.sum((active_bin_centers[:, 0] > 0) & (active_bin_centers[:, 1] < 0))

    # --- Step 3: Calculate and print percentages ---
    coverage_tr = (active_q_tr / total_q_tr * 100) if total_q_tr > 0 else 0
    coverage_tl = (active_q_tl / total_q_tl * 100) if total_q_tl > 0 else 0
    coverage_bl = (active_q_bl / total_q_bl * 100) if total_q_bl > 0 else 0
    coverage_br = (active_q_br / total_q_br * 100) if total_q_br > 0 else 0
    total_coverage = (total_active_bins / total_possible_bins * 100) if total_possible_bins > 0 else 0

    print(f"\nTotal Possible Bins: {total_possible_bins}")
    print(f"Total Active Bins: {total_active_bins}")
    print(f"Total Coverage: {total_coverage:.2f}%\n")

    print("Coverage per Quadrant:")
    print(f"  - Top-Left (Authoritarian-Left):   {coverage_tl:.2f}% ({active_q_tl}/{total_q_tl})")
    print(f"  - Top-Right (Authoritarian-Right): {coverage_tr:.2f}% ({active_q_tr}/{total_q_tr})")
    print(f"  - Bottom-Right (Libertarian-Right):{coverage_br:.2f}% ({active_q_br}/{total_q_br})")
    print(f"  - Bottom-Left (Libertarian-Left):  {coverage_bl:.2f}% ({active_q_bl}/{total_q_bl})")
    print('-' * 70)
    

# ======================================= MAIN =======================================
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize political compass results from LLMs.")
    parser.add_argument(
        '--use_chat_template', 
        type=bool,
        default=True,
        help="Use chat template for prompts (True/False)."
    )

    parser.add_argument(
        '--model', 
        type=int,
        default=2,
        help="The model to combine batches for."
    )    

    parser.add_argument(
        "--action", 
        type=str, 
        default='plot',
        choices=['plot', 'stats'], 
        help="Action to perform: 'plot' or 'stats'.")

    parser.add_argument(
        "--color",
        type=str,
        default='green',
        choices=['red', 'green', 'blue', 'yellow'],
        help="Color scheme for the plots (red/green/blue/yellow)."
    )
    
    args = parser.parse_args()

    USE_CHAT_TEMPLATE = args.use_chat_template
    SELECTED_MODEL = args.model
    ACTION = args.action
    COLOR = args.color

    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3", # 0
        "meta-llama/Llama-3.1-8B-Instruct", # 1
        "Qwen/Qwen2.5-7B-Instruct", # 2
        "HuggingFaceH4/zephyr-7b-beta", # 3
        "Qwen/Qwen2.5-32B-Instruct", # 4
        "meta-llama/Llama-3.1-70B-Instruct", # 5
        "meta-llama/Llama-3.3-70B-Instruct", # 6
        "Qwen/Qwen2.5-72B-Instruct" # 7
        ]
        
    MODEL = MODELS[SELECTED_MODEL]
    MODEL = MODEL.split('/')[-1]
    
    SETTINGS = ["base", "right", "left"] # Defines the settings to try and load

    GRIDSIZE = 30
    
    print()
    print("="*70)
    if ACTION == 'plot':
        print(f"--- Generating plots for model: {MODEL} ---")

        # --- Parameters to control plotting, easily modifiable here ---
        plot_params = {
            'bins': 'log',
            'label_gap': 2.8,
            'label_margin': 0.6,
            'marginal_ratio': 3,
            'tick_interval': 5,
            'label_size': 31,
            'tick_size': 28,
            'colorbar_size': 28,
            'marginal_alpha': 0.4,
            'origin_lines_alpha': 0.4,
            'color': COLOR
        }
        # ---

        df_base = load_compass_data(MODEL, 'base', USE_CHAT_TEMPLATE)
        if df_base is None:
            print("Cannot proceed with plotting as 'base' data is missing.")
            return # Exit if base data is not available for plotting.

        base_centroid = calculate_centroid(df_base)
        print(f"Base Centroid: {base_centroid}")
        print("="*70)
        print()

        # Load no-persona compass data
        no_persona_position = load_no_persona_compass_data(MODEL, USE_CHAT_TEMPLATE)
        if no_persona_position is not None:
            print(f"No-persona LLM position: {no_persona_position}")
        else:
            print("No-persona LLM position data not found")
        
        print("="*70)
        print()

        for setting in SETTINGS:
            print(f"\nProcessing setting: {setting}")
            df_setting = load_compass_data(MODEL, setting, USE_CHAT_TEMPLATE)

            if df_setting is None: # Check if data was loaded successfully
                continue # Skip to the next setting if data is missing

            # Prepare plot-specific parameters
            is_base_case = (setting == 'base')
            dot_to_plot = None if is_base_case else base_centroid
            use_triangle = not is_base_case

            # Only show circle for base setting and if no-persona data is available
            circle_to_plot = no_persona_position if is_base_case and no_persona_position is not None else None

            output_path = os.path.join(PROJ_ROOT, 'images', MODEL, setting, f'compass_density_log_{MODEL}_{setting}.png')
            
            # Create the plot using parameters defined above
            g = plot_compass_density(
                df_setting,
                gridsize=GRIDSIZE,
                np_dot=dot_to_plot,
                triangle=use_triangle,
                circle_dot=circle_to_plot,
                **plot_params # Unpack the main dictionary of plot parameters
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            g.savefig(output_path, bbox_inches='tight')
            
            print(f"Plot saved to {output_path}")
            plt.close(g.figure)            

    elif ACTION == 'stats':
        print(f"--- Running significance tests for model: {MODEL} ---")

        # Load data for all required settings
        df_base = load_compass_data(MODEL, 'base', USE_CHAT_TEMPLATE)
        df_right = load_compass_data(MODEL, 'right', USE_CHAT_TEMPLATE)
        df_left = load_compass_data(MODEL, 'left', USE_CHAT_TEMPLATE)

        # Centroid analysis
        print()
        print("="*70)
        print("--- Centroid Analysis ---")
        if df_left is not None:
            avg_dist_left = run_centroid_analysis(df_left)
            print(f"Average distance from centroid (left): {avg_dist_left:.4f}")
        else:
            print("Skipping centroid analysis for 'left' due to missing data.")

        if df_base is not None:
            avg_dist_base = run_centroid_analysis(df_base)
            print(f"Average distance from centroid (base): {avg_dist_base:.4f}")
        else:
            print("Skipping centroid analysis for 'base' due to missing data.")

        if df_right is not None:
            avg_dist_right = run_centroid_analysis(df_right)
            print(f"Average distance from centroid (right): {avg_dist_right:.4f}")
        else:
            print("Skipping centroid analysis for 'right' due to missing data.")
        print("="*70)
        print()

        # Movement analysis
        print()
        print("="*70)
        print("="*70)
        print("--- Movement Analysis ---")

        if df_base is not None and df_right is not None:
            print()
            print("BASE wrt RIGHT")
            print("-"*70)
            run_significance_test(df_base, df_right)
            print()
        else:
            print("Skipping 'BASE wrt RIGHT' significance test due to missing data.")

        if df_base is not None and df_left is not None:
            print("BASE wrt LEFT")
            print("-"*70)
            run_significance_test(df_base, df_left)
        else:
            print("Skipping 'BASE wrt LEFT' significance test due to missing data.")

        # Coverage Analysis
        print()
        print("="*70)
        print("="*70)
        print("--- Coverage Analysis ---")

        if df_left is not None:
            run_coverage_analysis(df_left, GRIDSIZE, 'left')
        else:
            print("Skipping coverage analysis for 'left' due to missing data.")

        if df_base is not None:
            run_coverage_analysis(df_base, GRIDSIZE, 'base')
        else:
            print("Skipping coverage analysis for 'base' due to missing data.")

        if df_right is not None:
            run_coverage_analysis(df_right, GRIDSIZE, 'right')
        else:
            print("Skipping coverage analysis for 'right' due to missing data.")

    else:
        raise ValueError("Invalid action specified. Use 'plot' or 'stats'.")
# ======================================= MAIN =======================================


if __name__ == "__main__":
    main()