import pandas as pd
import json
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.interpolate import griddata

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

MODELS = [
    "Mistral-7B-Instruct-v0.3", # 0
    "Llama-3.1-8B-Instruct", # 1
    "Qwen2.5-7B-Instruct", # 2
    "zephyr-7b-beta", # 3
    "Llama-3.1-70B-Instruct", # 4
    "Llama-3.3-70B-Instruct", # 5
    "Qwen2.5-72B-Instruct" # 6
]

CLUSTERED_PERSONA_PATH_BEG = "../../data/processed/clustered_persona/"
RESULTS_BASE_DIR = "../../results"
SAVE_BASE_DIR = "../../images"



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merge persona cluster data with political compass data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=int,
        default=0,
        choices=range(len(MODELS)),
        help=f"The index of the model to use. 0: {MODELS[0]}, 1: {MODELS[1]}, etc."
    )
    parser.add_argument(
        '--persona',
        type=str,
        default='base',
        choices=['base', 'left', 'right'],
        help="The configuration of the persona descriptions used for inference."
    )
    parser.add_argument(
        '--topics',
        type=int,
        default=15,
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose output for debugging purposes."
    )
    parser.add_argument(
        '--legend',
        type=bool,
        default=False,
    )
    return parser.parse_args()


def load_clustered_personas(file_path, verbose=False):
    print(f"Loading clustered persona data from: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    # The 'topic_keywords_with_scores' column is stored as a JSON string
    if 'topic_keywords_with_scores' in df.columns:
        df['topic_keywords_with_scores'] = df['topic_keywords_with_scores'].apply(json.loads)

    if verbose:
        print("\n" + "="*30 + " Clustered Persona DataFrame " + "="*30)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Print a sample of a complex column
        if 'topic_keywords_with_scores' in df.columns and not df.empty:
             print("\nSample 'topic_keywords_with_scores' entry:")
             print(df.iloc[0]['topic_keywords_with_scores'])

        # Print unique topics if the column exists
        if 'topic_keywords' in df.columns:
            # Convert numpy arrays to tuples to make them hashable for unique()
            topic_keywords_as_tuples = df['topic_keywords'].apply(
                lambda x: tuple(x) if isinstance(x, np.ndarray) else x
            )
            unique_topics = topic_keywords_as_tuples.unique()
            print("\nUnique topics found:")
            for topic in unique_topics:
                print(f"- {topic}")
        print("="*80 + "\n")

    return df


def load_compass_data(base_dir, model_name, persona_config, verbose=False):
    file_path = f"{base_dir}/{model_name}/{persona_config}/final_compass.pqt"
    print(f"Loading compass data from: {file_path}")
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        print("Please ensure you have run the inference for this model and configuration.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print("\n" + "="*30 + " Compass Persona DataFrame " + "="*30)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("="*80 + "\n")

    return df


# ================================
# Statistical Deviation Maps
# ================================
def plot_statistical_deviation_map(df_background, df_foreground, topic_keywords, model_name, persona_config, output_path, config):
    gridsize = config['gridsize']
    x_bg = [pos[0] for pos in df_background['compass_position']]
    y_bg = [pos[1] for pos in df_background['compass_position']]
    x_fg = [pos[0] for pos in df_foreground['compass_position']]
    y_fg = [pos[1] for pos in df_foreground['compass_position']]

    if len(x_fg) == 0:
        print(f"Skipping plot for topic '{', '.join(topic_keywords)}' as it has no data points.")
        return

    # --- Step 1: Get hex bin counts for both distributions ---
    # A temporary figure is used to calculate the bin counts without displaying it.
    temp_fig, temp_ax = plt.subplots()
    bg_hex = temp_ax.hexbin(x_bg, y_bg, gridsize=gridsize, extent=(-10, 10, -10, 10), visible=False)
    counts_bg = bg_hex.get_array()
    fg_hex = temp_ax.hexbin(x_fg, y_fg, gridsize=gridsize, extent=(-10, 10, -10, 10), visible=False)
    counts_fg = fg_hex.get_array()
    plt.close(temp_fig)

    # --- Step 2: Calculate expected counts and z-scores ---
    total_bg = np.sum(counts_bg)
    total_fg = np.sum(counts_fg)

    if total_bg == 0 or total_fg == 0:
        print(f"Warning: No background or foreground data for topic '{', '.join(topic_keywords)}'")
        return

    expected_counts = total_fg * (counts_bg / total_bg)
    p_bg = counts_bg / total_bg
    variance = total_fg * p_bg * (1 - p_bg)
    
    min_expected = config['min_expected_count']
    mask = expected_counts >= min_expected
    z_scores = np.zeros_like(expected_counts)
    z_scores[mask] = (counts_fg[mask] - expected_counts[mask]) / np.sqrt(variance[mask])
    z_scores[~mask] = 0

    # --- Step 3: Create the statistical deviation plot ---
    fig, ax = plt.subplots(figsize=config['figsize'])
    
    final_hex = ax.hexbin(x_bg, y_bg, gridsize=gridsize, extent=(-10, 10, -10, 10),
                         cmap='coolwarm', mincnt=0)
    final_hex.set_array(z_scores)

    # Centroid
    if config.get('show_population_centroid', False):
        if x_bg and y_bg:  # Check if there is background data
            # Calculate centroid from the background population coordinates
            centroid_x = np.mean(x_bg)
            centroid_y = np.mean(y_bg)
            print(f"      - Population Centroid: ({centroid_x:.2f}, {centroid_y:.2f})")

            # Plot the large black triangle (outline)
            ax.scatter(
                centroid_x, centroid_y,
                marker='^',
                color=config['centroid_marker_color_outer'],
                s=500,
                zorder=config['centroid_marker_zorder'] - 1 # Draw outline first
            )
            # Plot the smaller white triangle on top
            ax.scatter(
                centroid_x, centroid_y,
                marker='^',
                color=config['centroid_marker_color_inner'],
                s=358,
                zorder=config['centroid_marker_zorder'] # Draw inner part on top
            )
        else:
            print("      - Warning: Cannot calculate centroid, background data is empty.")

    # Background outline
    if config.get('show_population_outline', False):
            hex_centers = final_hex.get_offsets()
            if len(hex_centers) > 0 and len(counts_bg) == len(hex_centers):
                try:
                    xi = np.linspace(-10, 10, 200)
                    yi = np.linspace(-10, 10, 200)
                    Xi, Yi = np.meshgrid(xi, yi)
                    zi_bg = griddata(hex_centers, counts_bg, (Xi, Yi), method='linear', fill_value=0)

                    ax.contour(Xi, Yi, zi_bg,
                            levels=[1], 
                            colors=config['population_outline_color'],
                            linewidths=config['population_outline_linewidth'],
                            alpha=config['population_outline_alpha'])
                except Exception as e:
                    print(f"Warning: Could not add background population outline: {e}")

    max_z = max(config['min_z_score_clip'], np.percentile(np.abs(z_scores[mask]), config['z_score_clip_percentile']))
    final_hex.set_clim(-max_z, max_z)

    # --- Step 4: Add statistical significance contours ---
    hex_centers = final_hex.get_offsets()
    if len(hex_centers) > 0 and len(z_scores) == len(hex_centers):
        try:
            xi = np.linspace(-10, 10, 100)
            yi = np.linspace(-10, 10, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = griddata(hex_centers, z_scores, (Xi, Yi), method='linear', fill_value=0)
            ax.contour(Xi, Yi, np.abs(zi),
                       levels=config['contour_levels'],
                       colors=config['contour_colors'],
                       linewidths=config['contour_linewidths'],
                       linestyles=config['contour_linestyles'],
                       alpha=config['contour_alpha'])
        except Exception as e:
            print(f"Warning: Could not add significance contours: {e}")

    # --- Step 5: Formatting (using config) ---
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='box')

    ticks = [-10, -5, 0, 5, 10]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Create string labels, but set the first one (for -10) to be empty
    x_tick_labels = [str(t) for t in ticks]
    x_tick_labels[0] = ''
    y_tick_labels = [str(t) for t in ticks]
    y_tick_labels[0] = ''
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    # Manually add a single '-10' text label in the corner with a slight offset
    ax.text(-10.2, -10.2, '-10', ha='right', va='top', fontsize=config['tick_fontsize'])

    ax.tick_params(axis='both', which='major', labelsize=config['tick_fontsize'])

    ax.grid(True, which='major', linestyle='-', alpha=config['grid_alpha'])
    ax.axhline(0, color='black', alpha=config['axis_alpha'], lw=config['axis_linewidth'])
    ax.axvline(0, color='black', alpha=config['axis_alpha'], lw=config['axis_linewidth'])

    label_gap = config['text_label_gap']
    label_pos_y = -10 - label_gap
    label_pos_x = -10 - label_gap
    ax.text(-10, label_pos_y, '← Left', ha='left', va='top', fontsize=config['annotation_fontsize'])
    ax.text(10, label_pos_y, 'Right →', ha='right', va='top', fontsize=config['annotation_fontsize'])
    ax.text(label_pos_x, 10, 'Authorit. →', ha='right', va='top', rotation=90, fontsize=config['annotation_fontsize'])
    ax.text(label_pos_x, -10, '← Libert.', ha='right', va='bottom', rotation=90, fontsize=config['annotation_fontsize'])

    cbar = fig.colorbar(final_hex, ax=ax, fraction=0.046, pad=0.04)
    if config['use_cbar_label']:
        cbar.set_label(
            'Statistical Deviation (Z-score)\n(Red = Over-represented, Blue = Under-represented)',
            size=config['cbar_label_fontsize'],
            labelpad=config['cbar_label_padding'],      # Use parameter for gap
            y=config['cbar_label_y_position']           # Use parameter for vertical position
        )
    else:
        cbar.set_label(
            ' \n ',
            size=config['cbar_label_fontsize'],
            labelpad=config['cbar_label_padding'],      # Use parameter for gap
            y=config['cbar_label_y_position']           # Use parameter for vertical position
        )
    cbar.ax.tick_params(labelsize=config['tick_fontsize'])

    # Add significance level markers on colorbar
    # Add significance level markers on colorbar
    for i, level in enumerate(config['contour_levels']):
        color = config['contour_colors'][i] if i < len(config['contour_colors']) else 'black'
        linestyle = config['contour_linestyles'][i] if i < len(config['contour_linestyles']) else '--'
        linewidth = config['contour_linewidths'][i] if i < len(config['contour_linewidths']) else config['cbar_line_width']
        cbar.ax.axhline(level, color=color, linestyle=linestyle, alpha=config['contour_alpha'], linewidth=linewidth)
        cbar.ax.axhline(-level, color=color, linestyle=linestyle, alpha=config['contour_alpha'], linewidth=linewidth)
    
    # topic_title = ", ".join(topic_keywords)
    # fig.suptitle(f"Statistical Deviation Map: {model_name} ({persona_config})\n"
    #             f"Topic: {topic_title} (n={total_fg} vs N={total_bg})",
    #             fontsize=config['title_fontsize'], y=0.98)
    
    # fig.text(0.02, 0.02,
    #          f"Z-scores show statistical significance of concentration differences.\n"
    #          f"Contours at |Z|={config['contour_levels']}. Bins with expected count < {min_expected} are masked.",
    #          fontsize=config['footnote_fontsize'], alpha=config['footnote_alpha'], transform=fig.transFigure)

    

    fig.savefig(output_path, bbox_inches='tight', dpi=config['dpi'])
    plt.close(fig)

    # Print summary
    significant_bins = np.sum(np.abs(z_scores) > 2)
    highly_significant_bins = np.sum(np.abs(z_scores) > 3)
    total_bins_analyzed = np.sum(mask)
    
    print(f"    Statistical summary for '{', '.join(topic_keywords)}':")
    print(f"      - Bins analyzed: {total_bins_analyzed}")
    print(f"      - Significant bins (|z|>2): {significant_bins}")
    print(f"      - Highly significant bins (|z|>3): {highly_significant_bins}")
    print(f"      - Max |z-score|: {np.max(np.abs(z_scores)):.2f}")


def create_statistical_deviation_plots(merged_df, model_name, persona_config, config):
    merged_df['topic_tuple'] = merged_df['topic_keywords'].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
    unique_topics = merged_df['topic_tuple'].unique()

    output_dir = os.path.join(SAVE_BASE_DIR, model_name, persona_config, "statistical_deviation_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving {len(unique_topics)} statistical deviation plots to: {output_dir}")

    for i, topic_tuple in enumerate(unique_topics):
        topic_name_safe = topic_tuple[0].replace(" ", "-").lower()
        print(f"  ({i+1}/{len(unique_topics)}) Creating statistical deviation map for: {', '.join(topic_tuple)}")

        df_topic = merged_df[merged_df['topic_tuple'] == topic_tuple]

        plot_filename = f"compass_zscore_{topic_name_safe}.png"
        output_path = os.path.join(output_dir, plot_filename)

        # Pass the config dictionary to the plotting function
        plot_statistical_deviation_map(
            df_background=merged_df,
            df_foreground=df_topic,
            topic_keywords=topic_tuple,
            model_name=model_name,
            persona_config=persona_config,
            output_path=output_path,
            config=config
        )
    print("All statistical deviation plots generated successfully.")



def main():
    args = parse_arguments()
    CLUSTERED_PERSONA_PATH = f"{CLUSTERED_PERSONA_PATH_BEG}cleaned_persona_clustered_{args.topics}topics.pqt"
    PLOT_CONFIG = {
        # Figure and Axes
        'figsize': (10, 8.5),
        'dpi': 150,

        # Fonts
        'title_fontsize': 14,
        'tick_fontsize': 28,
        'annotation_fontsize': 28,
        'cbar_label_fontsize': 20,
        'footnote_fontsize': 8,

        # Lines and Grid
        'grid_alpha': 0.3,
        'axis_linewidth': 1.5,
        'axis_alpha': 0.6,

        # Hexbin plot parameters
        'gridsize': 30,

        # Statistical parameters
        'min_expected_count': 5,
        'z_score_clip_percentile': 95,
        'min_z_score_clip': 3.0,

        # Contour plot parameters
        # 'contour_levels': [2, 3],
        'contour_levels': [3.3, 5], # z-score levels for significance
        'contour_colors': ['black', 'lime'],
        'contour_linewidths': [2.0, 3.5],
        'contour_linestyles': ['--', '-'],
        'contour_alpha': 0.7,

        # Colorbar
        'use_cbar_label': False,
        'cbar_line_width': 1.0,
        'cbar_label_padding': 20,          # Gap between colorbar and its label text.
        'cbar_label_y_position': 0.45,      # Vertical position of the label (0.0=bottom, 0.5=center, 1.0=top).

        # Text and Annotations
        'text_label_gap': 2,
        'footnote_alpha': 0.7,

        # Population outline
        'show_population_outline': True,
        'population_outline_color': 'black',           # Color of the outline
        'population_outline_linewidth': 1.0,           # A thin but clearly visible line width
        'population_outline_alpha': 0.9,               # Make it mostly opaque

        # Centroid marker configuration
        'show_population_centroid': True,
        'centroid_marker_size_outer': 270,         # Size of the black background triangle
        'centroid_marker_size_inner': 210,         # Size of the white foreground triangle
        'centroid_marker_color_outer': 'black',
        'centroid_marker_color_inner': 'white',
        'centroid_marker_zorder': 5,               # Ensures the marker is drawn on top
    }
    
    selected_model_name = MODELS[args.model]
    print(f"Selected Model: {selected_model_name}")
    print(f"Persona Configuration: {args.persona}\n")

    # 1. Load the two DataFrames
    cluster_df = load_clustered_personas(CLUSTERED_PERSONA_PATH, verbose=args.verbose)
    compass_df = load_compass_data(
        RESULTS_BASE_DIR,
        selected_model_name,
        args.persona,
        verbose=args.verbose
    )

    # 2. Join the DataFrames
    print("Joining DataFrames on shared attribute...")
    join_keys = ['persona', 'persona_id', 'language', 'cleaned_persona']
    merged_df = pd.merge(cluster_df, compass_df, on=join_keys, how='inner')
    # Drop statement_id, statement, prompt, response and int_stance from the merged_df
    merged_df = merged_df.drop(columns=['statement_id', 'statement', 'prompt', 'response', 'int_stance'])
    # Drop duplicates
    merged_df = merged_df.drop_duplicates(subset=['persona_id'], keep='first', ignore_index=True)

    print("Join complete.")

    print("\n" + "="*30 + " Merged DataFrame Summary " + "="*30)
    print(f"Shape of merged data: {merged_df.shape}")
    print(f"Columns: {merged_df.columns.tolist()}")
    print("="*80 + "\n")

    # 3. Create and save plots for each topic
    create_statistical_deviation_plots(
        merged_df,
        selected_model_name,
        args.persona,
        config=PLOT_CONFIG
    )

    print("\nScript finished successfully.")


if __name__ == "__main__":
    main()