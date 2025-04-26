# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Any, Optional, List, Tuple
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# Removed PIL import

# --- Streamlit App Layout ---
st.set_page_config(page_title="Water Quality Clustering Dashboard")

# --- Configuration & Constants ---
DATA_DIR = "streamlit_app_data"
# Make ORIGINAL_DATA_PATH relative to the app.py location
ORIGINAL_DATA_PATH = os.path.join("MachineLearningAssignment", "data", "processed", "processed_waterPollution.csv")

# Files expected in DATA_DIR
SCORES_FILE = os.path.join(DATA_DIR, "optimized_model_scores.csv")
PARAMS_FILE = os.path.join(DATA_DIR, "optimized_model_params.joblib")
PCA_DATA_FILE = os.path.join(DATA_DIR, "pca_2D_data.csv") # Focusing on 2D as used in tuning
PCA_OBJECT_FILE = os.path.join(DATA_DIR, "pca_2D_object.joblib")

# Key features used for radar chart (match your notebook)
KEY_FEATURES_RADAR = [
    'resultMeanValue_log1p', 'TerraMarineProtected_2016_2018',
    'PopulationDensity_log1p', 'gdp_log1p', # Assuming gdp_log1p is correct name in file
    'waste_treatment_recycling_percent',
    'composition_food_organic_waste_percent', 'composition_plastic_percent'
]

# --- Utility Functions ---

@st.cache_data
def load_csv(filepath: str) -> Optional[DataFrame]:
    """Loads a CSV file into a Pandas DataFrame."""
    abs_filepath = os.path.abspath(filepath)
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at specified path: {filepath} (Absolute: {abs_filepath})")
        return None
    except Exception as e:
        st.error(f"An error occurred loading {filepath}: {e}")
        return None

@st.cache_data
def load_joblib(filepath: str) -> Any:
    """Loads a joblib file."""
    abs_filepath = os.path.abspath(filepath)
    try:
        obj = joblib.load(filepath)
        return obj
    except FileNotFoundError:
        st.error(f"Error: File not found at specified path: {filepath} (Absolute: {abs_filepath})")
        return None
    except Exception as e:
        st.error(f"An error occurred loading {filepath}: {e}")
        return None

@st.cache_data
def load_npy(filepath: str) -> Optional[ndarray]:
    """Loads a NumPy .npy file."""
    abs_filepath = os.path.abspath(filepath)
    try:
        arr = np.load(filepath)
        return arr
    except FileNotFoundError:
        st.error(f"Error: File not found at specified path: {filepath} (Absolute: {abs_filepath})")
        return None
    except Exception as e:
        st.error(f"An error occurred loading {filepath}: {e}")
        return None

@st.cache_data
def get_available_models_and_labels(data_dir: str) -> Dict[str, Optional[ndarray]]:
    """Finds available model label files and loads them."""
    model_labels = {}
    abs_data_dir = os.path.abspath(data_dir)
    try:
        if not os.path.isdir(abs_data_dir):
            st.error(f"Data directory not found: {data_dir} (Absolute: {abs_data_dir})")
            return model_labels

        for filename in os.listdir(data_dir):
            if filename.startswith("best_labels_") and filename.endswith("_2D.npy"):
                parts = filename.replace("best_labels_", "").replace("_2D.npy", "")
                model_name = parts
                filepath = os.path.join(data_dir, filename)
                labels = load_npy(filepath)
                if labels is not None:
                    model_labels[model_name] = labels
                else:
                     st.warning(f"Could not load labels for {model_name} from {filename}")
    except Exception as e:
        st.error(f"Error reading data directory {data_dir}: {e}")
    return model_labels

def generate_cluster_explanations(
    cluster_means: DataFrame,
    cluster_means_norm: DataFrame,
    top_countries_data: Dict[int, Dict[str, int]],
    feature_list: List[str]
) -> Dict[int, str]:
    """
    Generates textual explanations for each cluster based on its profile.

    Args:
        cluster_means: DataFrame of mean values per cluster (original scale).
        cluster_means_norm: DataFrame of normalized mean values per cluster (0-1 scale).
        top_countries_data: Dict where keys are cluster IDs and values are dicts
                             of {Country: Count} for top countries.
        feature_list: List of features used in the profile (columns of the means DFs).

    Returns:
        A dictionary where keys are cluster IDs and values are explanation strings.
    """
    explanations = {}
    num_clusters = len(cluster_means)
    num_features = len(feature_list)

    # Define thresholds for 'high' and 'low' based on normalized scale
    high_threshold = 0.70 # Top 30%
    low_threshold = 0.30  # Bottom 30%

    for cluster_id in cluster_means.index:
        desc = f"**Cluster {cluster_id}:**\n"
        high_features = []
        low_features = []
        norm_profile = cluster_means_norm.loc[cluster_id]
        orig_profile = cluster_means.loc[cluster_id]

        for feature in feature_list:
            norm_val = norm_profile[feature]
            if norm_val >= high_threshold:
                high_features.append((feature, orig_profile[feature]))
            elif norm_val <= low_threshold:
                low_features.append((feature, orig_profile[feature]))

        # Sort features by normalized value for potentially better descriptions
        high_features.sort(key=lambda item: cluster_means_norm.loc[cluster_id, item[0]], reverse=True)
        low_features.sort(key=lambda item: cluster_means_norm.loc[cluster_id, item[0]])

        if high_features:
            desc += f"*   Primarily characterized by **high** values in: "
            desc += ", ".join([f"{f} ({v:.2f})" for f, v in high_features[:3]]) # Show top 3 high
            if len(high_features) > 3: desc += " and others."
            desc += ".\n"
        else:
             desc += "*   Does not exhibit distinctly high values in the key features compared to other clusters.\n"


        if low_features:
            desc += f"*   Also shows relatively **low** values in: "
            desc += ", ".join([f"{f} ({v:.2f})" for f, v in low_features[:3]]) # Show top 3 low
            if len(low_features) > 3: desc += " and others."
            desc += ".\n"
        else:
             desc += "*   Does not exhibit distinctly low values in the key features compared to other clusters.\n"

        # Add country information
        if cluster_id in top_countries_data and top_countries_data[cluster_id]:
            top_countries_str = ", ".join([f"{country} ({count})" for country, count in top_countries_data[cluster_id].items()])
            desc += f"*   Predominantly includes records from: **{top_countries_str}**.\n"
        else:
            desc += "*   Country distribution information not available or cluster has few records.\n"

        explanations[cluster_id] = desc

    return explanations

# --- Plotting Functions ---
def plot_country_distribution_per_cluster(df_profiled_data: DataFrame, cluster_col: str, top_n: int = 3) -> Optional[plt.Figure]:
    """
    Generates faceted bar plots showing the top N countries for each cluster.

    Args:
        df_profiled_data: DataFrame containing original data merged with cluster labels.
                          Must include 'Country' column and the specified cluster_col.
        cluster_col: The name of the column containing the cluster labels.
        top_n: The number of top countries to display for each cluster.

    Returns:
        A Matplotlib Figure object containing the plots, or None if plotting fails.
    """
    if 'Country' not in df_profiled_data.columns:
        st.warning("Cannot plot country distribution: 'Country' column not found.")
        return None
    if cluster_col not in df_profiled_data.columns:
        st.warning(f"Cannot plot country distribution: Cluster column '{cluster_col}' not found.")
        return None

    # Exclude noise points (-1) and get unique cluster IDs
    cluster_ids = sorted([cid for cid in df_profiled_data[cluster_col].unique() if cid != -1 and not pd.isna(cid)])

    if not cluster_ids:
        st.info("No non-noise clusters found to plot country distribution.")
        return None

    # Determine grid size for subplots
    n_clusters = len(cluster_ids)
    n_cols = 2 # Adjust number of columns for layout if needed
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False) # Ensure axes is always 2D array
    axes_flat = axes.flatten() # Flatten for easy iteration

    all_top_countries_data = {} # To store data for explanations

    for i, cluster_id in enumerate(cluster_ids):
        ax = axes_flat[i]
        cluster_data = df_profiled_data[df_profiled_data[cluster_col] == cluster_id]

        if cluster_data.empty:
            ax.set_title(f"Cluster {cluster_id} (No Data)")
            ax.text(0.5, 0.5, 'No data points in cluster', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        country_counts = cluster_data['Country'].value_counts().nlargest(top_n)

        if country_counts.empty:
            ax.set_title(f"Cluster {cluster_id} (No Country Data)")
            ax.text(0.5, 0.5, 'No country data for cluster', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        # Store for explanations
        all_top_countries_data[cluster_id] = country_counts.to_dict()

        # Create bar plot
        sns.barplot(x=country_counts.index, y=country_counts.values, ax=ax, palette="viridis")
        ax.set_title(f"Cluster {cluster_id} - Top {top_n} Countries")
        ax.set_ylabel("Number of Records")
        ax.set_xlabel("Country")
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    return fig, all_top_countries_data # Return figure and the data dictionary

def plot_pca_clusters(X_pca: ndarray, labels: ndarray, pca_obj: PCA, model_name: str) -> None:
    """Generates a 2D scatter plot of PCA results colored by cluster labels."""
    if X_pca is None or labels is None or pca_obj is None:
        st.warning("Missing data for PCA plot.")
        return

    unique_labels = np.unique(labels)
    unique_labels_arr = np.atleast_1d(unique_labels)
    n_clusters = np.sum(unique_labels_arr != -1)

    if n_clusters == 0 and -1 not in unique_labels_arr: # Added check if NO clusters and NO noise
         st.warning("Plotting skipped: No clusters or noise points found in labels.")
         return
    elif n_clusters == 0 and -1 in unique_labels_arr:
         st.warning("Plotting skipped: Only noise points found.")
         # Optionally, could plot only noise points if desired
         # return # Currently returning

    has_noise = -1 in unique_labels_arr

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis

    # --- **FIX APPLIED HERE** ---
    # Create mapping from actual cluster label (0, 1, ...) to color index (0, 1, ...)
    label_indices = {}
    color_idx_counter = 0
    for label in unique_labels_arr:
        if label != -1:
            label_indices[label] = color_idx_counter
            color_idx_counter += 1

    # Generate colors only for the actual clusters found (size = n_clusters)
    # Ensure linspace has at least 1 segment even if n_clusters is 1
    colors = cmap(np.linspace(0, 1, max(1, n_clusters)))
    # --- End of Fix ---

    for k in unique_labels_arr:
        class_member_mask = (labels == k)
        # Check if any points belong to this cluster/noise label
        if not np.any(class_member_mask):
            continue # Skip if no points have this label

        xy = X_pca[class_member_mask]

        if k == -1:
            color_val = [0.5, 0.5, 0.5, 0.6] # Grey
            markersize = 4
            plot_label = 'Noise'
            edgecolor = 'none'
        else:
            # Use the correctly mapped color index
            cluster_idx = label_indices.get(k) # Should always find the key now
            if cluster_idx is None: # Safety check, should not happen with new logic
                 st.warning(f"Could not find color index for cluster label {k}. Skipping.")
                 continue
            # Handle case of 1 cluster (n_clusters=1 -> colors is array of size 1)
            color_val = colors[cluster_idx] if n_clusters > 0 else [0.0, 0.0, 0.0, 1.0] # Default black if error

            markersize = 6
            plot_label = f'Cluster {k}'
            edgecolor = 'k'

        ax.scatter(xy[:, 0], xy[:, 1], s=markersize*5, c=[color_val],
                   label=plot_label, alpha=0.7, edgecolor=edgecolor, linewidth=0.5)

    ax.set_xlabel(f"Principal Component 1 ({pca_obj.explained_variance_ratio_[0]:.1%} Variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca_obj.explained_variance_ratio_[1]:.1%} Variance)")
    ax.set_title(f'Optimized {model_name} Clusters (2D PCA Projection)')

    handles, displayed_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(displayed_labels, handles))
    if len(by_label) < 15:
        ax.legend(by_label.values(), by_label.keys(), loc='best', title="Cluster ID / Noise")
    elif len(by_label) > 0 : # Only show caption if there's actually something to omit
        st.caption("Legend omitted due to large number of clusters/noise.")

    ax.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig)


def plot_radar_chart(df_profiles_norm: DataFrame, model_name: str, k: int) -> None:
    """Generates a radar chart for normalized cluster profiles."""
    if df_profiles_norm.empty:
        st.warning("Cannot generate radar chart: No profile data available.")
        return
    if k <= 0:
         st.warning(f"Cannot generate radar chart: Invalid number of clusters k={k}.")
         return

    labels = df_profiles_norm.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the plot

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    # Ensure linspace has at least 1 segment even if k is 1
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, k)))

    for i, idx in enumerate(df_profiles_norm.index): # Use enumerate to get color index
        values = df_profiles_norm.loc[idx].tolist()
        values += values[:1] # Close the plot
        color_val = colors[i] if k > 0 else colors[0] # Use index i
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {idx}', color=color_val)
        ax.fill(angles, values, color=color_val, alpha=0.2)

    ax.set_title(f"Normalized Cluster Profiles ({model_name}, k={k})", size=16, y=1.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{i:.1f}" for i in np.linspace(0, 1, 5)])
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1)) # Slightly adjust legend position
    ax.set_ylim(0, 1)
    st.pyplot(fig)


# --- Load Data at the Start ---
df_scores = load_csv(SCORES_FILE)
df_pca_2d = load_csv(PCA_DATA_FILE)
pca_2d_obj = load_joblib(PCA_OBJECT_FILE)
model_labels_dict = get_available_models_and_labels(DATA_DIR)
df_original_processed = load_csv(ORIGINAL_DATA_PATH)


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Performance", "Cluster Visualization", "Cluster Profiling", "Dataset Explorer"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    This dashboard visualizes the results of clustering analysis performed on water quality data,
    optimized using techniques from Part B of the assignment.
    """
)

# --- Page Content ---

if page == "Overview":
    st.title("Water Quality Clustering Analysis Dashboard")
    st.markdown("""
        Welcome to the interactive dashboard for exploring water quality clustering results.
        This application presents the findings from the analysis notebooks, focusing on the
        performance and characteristics of different clustering algorithms after hyperparameter tuning.

        **Navigate using the sidebar to explore:**
        *   **Model Performance:** Compare evaluation metrics for the optimized models.
        *   **Cluster Visualization:** View the clusters projected onto 2D PCA space.
        *   **Cluster Profiling:** Analyze the characteristics of each cluster using a radar chart.
        *   **Dataset Explorer:** Interactively filter and view the original processed dataset.

        The data used for clustering was dimensionality-reduced using Principal Component Analysis (PCA).
        The primary goal was to identify natural groupings within the water samples based on various features.
    """)


elif page == "Model Performance":
    st.title("Optimized Clustering Model Performance")
    st.markdown("Comparison of evaluation metrics for different clustering algorithms after hyperparameter tuning (Part B).")

    if df_scores is not None:
        metrics_higher_better = ['Silhouette', 'Calinski-Harabasz']
        metrics_lower_better = ['Davies-Bouldin']

        st.subheader("Performance Metrics Table")
        st.dataframe(df_scores.style.format({
            'Silhouette': '{:.3f}',
            'Davies-Bouldin': '{:.3f}',
            'Calinski-Harabasz': '{:.1f}',
            'Num_Clusters': '{:.0f}'
        }))

        st.subheader("Metrics Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Silhouette Score (Higher is Better)**")
            if 'Silhouette' in df_scores.columns:
                st.bar_chart(df_scores.set_index('Algorithm')['Silhouette'].dropna())
            else: st.caption("Silhouette scores not available.")

            st.markdown("**Calinski-Harabasz Score (Higher is Better)**")
            if 'Calinski-Harabasz' in df_scores.columns:
                st.bar_chart(df_scores.set_index('Algorithm')['Calinski-Harabasz'].dropna())
            else: st.caption("Calinski-Harabasz scores not available.")

        with col2:
            st.markdown("**Davies-Bouldin Score (Lower is Better)**")
            if 'Davies-Bouldin' in df_scores.columns:
                st.bar_chart(df_scores.set_index('Algorithm')['Davies-Bouldin'].dropna())
            else: st.caption("Davies-Bouldin scores not available.")

            st.markdown("**Number of Clusters Found**")
            if 'Num_Clusters' in df_scores.columns:
                st.bar_chart(df_scores.set_index('Algorithm')['Num_Clusters'].dropna())
            else: st.caption("Number of clusters not available.")

        st.subheader("Best Performing Model")
        metric_options = [col for col in df_scores.columns if col not in ['Algorithm', 'Num_Clusters']]
        if not metric_options:
             st.warning("No valid metrics available for ranking.")
        else:
            primary_metric = st.selectbox(
                "Select Primary Metric for Ranking:",
                options=metric_options,
                index=0
            )

            higher_is_better = primary_metric in metrics_higher_better
            df_sorted = df_scores.sort_values(by=primary_metric, ascending=(not higher_is_better), na_position='last')

            st.write(f"Models ranked by **{primary_metric}** (Best First):")
            display_cols = ['Algorithm', primary_metric]
            if 'Num_Clusters' in df_sorted.columns: display_cols.append('Num_Clusters')
            st.dataframe(
                df_sorted[display_cols].style.format({primary_metric: '{:.3f}', 'Num_Clusters': '{:.0f}'}),
                hide_index=True
            )

            best_model_name = df_sorted.iloc[0]['Algorithm'] if not df_sorted.empty else "N/A"
            best_score = df_sorted.iloc[0][primary_metric] if not df_sorted.empty and primary_metric in df_sorted.columns else "N/A"
            if isinstance(best_score, (int, float)):
                 st.success(f"🏆 Best model based on {primary_metric}: **{best_model_name}** (Score: {best_score:.3f})")
            else:
                 st.success(f"🏆 Best model based on {primary_metric}: **{best_model_name}** (Score: {best_score})")

    else:
        st.warning("Optimized model scores data (`optimized_model_scores.csv`) not found.")

elif page == "Cluster Visualization":
    st.title("Cluster Visualization (2D PCA)")
    st.markdown("Visualizing the clusters found by different optimized algorithms in the 2D PCA space.")

    if df_pca_2d is not None and pca_2d_obj is not None and model_labels_dict:
        available_models = sorted(list(model_labels_dict.keys()))

        if not available_models:
            st.warning("No optimized model label files (`best_labels_*.npy`) found in the data directory.")
        else:
            selected_model = st.selectbox("Select Model to Visualize:", options=available_models)

            if selected_model and selected_model in model_labels_dict:
                labels = model_labels_dict[selected_model]
                if labels is not None:
                     if len(labels) != len(df_pca_2d):
                          st.error(f"Length mismatch error: PCA data has {len(df_pca_2d)} rows, but labels for {selected_model} have {len(labels)} elements.")
                     else:
                          st.subheader(f"Visualization for: {selected_model}")
                          plot_pca_clusters(df_pca_2d.values, labels, pca_2d_obj, selected_model)
                else:
                     st.error(f"Labels for {selected_model} could not be loaded (returned None).")
            else:
                st.warning("Please select a valid model.")

    else:
        st.warning("PCA data (`pca_2D_data.csv`), PCA object (`pca_2D_object.joblib`), or model label files (`best_labels_*.npy`) not found. Cannot generate visualization.")


elif page == "Cluster Profiling":
    st.title("Cluster Profiling Analysis")
    st.markdown("Comparing the average characteristics of each cluster across key features, "
                "visualizing country distribution, and providing interpretations.")

    if df_original_processed is None:
         st.warning(f"Original processed dataset (`{ORIGINAL_DATA_PATH}`) could not be loaded. Cannot generate profiles.")
    elif not model_labels_dict:
         st.warning("No optimized model label files (`best_labels_*.npy`) found. Cannot generate profiles.")
    else:
        available_models = sorted([name for name, labels in model_labels_dict.items() if labels is not None])

        if not available_models:
            st.warning("No valid optimized model label files found for profiling.")
        else:
            selected_model_profile = st.selectbox("Select Model for Profiling:", options=available_models)

            if selected_model_profile and selected_model_profile in model_labels_dict:
                labels = model_labels_dict[selected_model_profile]
                if labels is not None:
                    df_profiling_data = df_original_processed.copy()
                    temp_cluster_col = f'_cluster_{selected_model_profile}'

                    if len(labels) == len(df_profiling_data):
                        df_profiling_data[temp_cluster_col] = labels
                        # Ensure labels are numeric, coercing errors to NaN
                        df_profiling_data[temp_cluster_col] = pd.to_numeric(df_profiling_data[temp_cluster_col], errors='coerce')
                        # Drop rows where cluster label assignment failed or was originally noise/invalid
                        df_profiling_data = df_profiling_data.dropna(subset=[temp_cluster_col])
                        # Convert labels to integer type after cleaning
                        df_profiling_data[temp_cluster_col] = df_profiling_data[temp_cluster_col].astype(int)

                        st.info(f"Assigned valid cluster labels to {len(df_profiling_data)} rows from {selected_model_profile}.")
                    else:
                        st.error(f"Length mismatch error: Original data has {len(df_original_processed)} rows, but labels for {selected_model_profile} have {len(labels)} elements. Cannot assign clusters for profiling.")
                        df_profiling_data = None # Prevent further processing

                    # --- Start Profiling if Data is Ready ---
                    if df_profiling_data is not None and not df_profiling_data.empty:

                        # --- 1. Country Distribution ---
                        st.subheader(f"Country Distribution per Cluster ({selected_model_profile})")
                        fig_country, top_countries_data = plot_country_distribution_per_cluster(df_profiling_data, temp_cluster_col, top_n=3)
                        if fig_country:
                            st.pyplot(fig_country)
                        else:
                            st.caption("Could not generate country distribution plot.")
                            top_countries_data = {} # Ensure it's initialized if plot fails


                        # --- 2. Feature Profiling (Radar Chart & Means Table) ---
                        st.subheader(f"Feature Profiles ({selected_model_profile})")
                        missing_radar_features = [f for f in KEY_FEATURES_RADAR if f not in df_profiling_data.columns]
                        if missing_radar_features:
                            st.warning(f"Following features specified for radar chart not found in original data: {missing_radar_features}")
                            available_radar_features = [f for f in KEY_FEATURES_RADAR if f in df_profiling_data.columns]
                        else:
                            available_radar_features = KEY_FEATURES_RADAR

                        if not available_radar_features:
                             st.error("None of the specified key features for radar chart were found in the data.")
                             cluster_means = pd.DataFrame() # Ensure empty df if no features
                        else:
                             # Calculate means, excluding noise (-1) if present
                             df_means_calc = df_profiling_data[df_profiling_data[temp_cluster_col] != -1]
                             if not df_means_calc.empty:
                                 cluster_means = df_means_calc.groupby(temp_cluster_col)[available_radar_features].mean()
                             else:
                                 cluster_means = pd.DataFrame() # Empty if only noise points

                        if cluster_means.empty:
                            st.warning("No valid non-noise clusters found to calculate feature profiles.")
                            cluster_means_norm_df = pd.DataFrame() # Ensure empty df
                        else:
                             # --- Radar Chart ---
                             scaler_radar = MinMaxScaler()
                             cluster_means_norm = scaler_radar.fit_transform(cluster_means)
                             cluster_means_norm_df = pd.DataFrame(cluster_means_norm, index=cluster_means.index, columns=cluster_means.columns)

                             num_clusters_found = len(cluster_means_norm_df)
                             plot_radar_chart(cluster_means_norm_df, selected_model_profile, num_clusters_found)

                             # --- Means Table ---
                             st.subheader("Mean Values per Cluster (Original Scale)")
                             st.dataframe(cluster_means.style.format("{:.2f}"))


                        # --- 3. Cluster Explanations ---
                        st.subheader(f"Cluster Interpretations ({selected_model_profile})")
                        if not cluster_means.empty and not cluster_means_norm_df.empty and top_countries_data is not None:
                             explanations = generate_cluster_explanations(
                                 cluster_means,
                                 cluster_means_norm_df,
                                 top_countries_data, # Pass the data from the country plot
                                 available_radar_features
                             )
                             for cluster_id in sorted(explanations.keys()):
                                 st.markdown(explanations[cluster_id])
                                 st.markdown("---") # Separator
                        else:
                             st.caption("Cannot generate interpretations - missing profile means, normalized means, or country data.")

                    # --- End Profiling Section ---
                    elif df_profiling_data is not None and df_profiling_data.empty:
                         st.warning("Data is empty after attempting to assign and clean cluster labels. Cannot proceed with profiling.")

                else:
                     st.error(f"Labels for {selected_model_profile} could not be loaded (returned None) for profiling.")
            else:
                st.warning("Please select a valid model.")

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown(f"Explore the original processed water quality dataset (`{ORIGINAL_DATA_PATH}`).")

    if df_original_processed is not None:
        st.info(f"Dataset shape: {df_original_processed.shape[0]} rows, {df_original_processed.shape[1]} columns")

        st.subheader("Filter Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'Country' in df_original_processed.columns:
                countries = sorted(df_original_processed['Country'].unique().astype(str))
                selected_countries = st.multiselect("Select Country:", options=countries, default=[])
            else:
                selected_countries = []
                st.caption("'Country' column not found.")

        with col2:
            if 'Start_Year' in df_original_processed.columns:
                 years_numeric = pd.to_numeric(df_original_processed['Start_Year'], errors='coerce').dropna()
                 if not years_numeric.empty:
                      min_year, max_year = int(years_numeric.min()), int(years_numeric.max())
                      if min_year < max_year:
                         selected_year_range = st.slider("Select Start Year Range:", min_year, max_year, (min_year, max_year))
                      else:
                         st.caption(f"Only one year available: {min_year}")
                         selected_year_range = (min_year, max_year)
                 else:
                      selected_year_range = None
                      st.caption("No valid numeric data in 'Start_Year'.")
            else:
                 selected_year_range = None
                 st.caption("'Start_Year' column not found.")

        with col3:
             poll_col = 'resultMeanValue_log1p'
             if poll_col in df_original_processed.columns:
                  poll_numeric = pd.to_numeric(df_original_processed[poll_col], errors='coerce').dropna()
                  if not poll_numeric.empty:
                       min_poll, max_poll = float(poll_numeric.min()), float(poll_numeric.max())
                       if min_poll < max_poll:
                            selected_poll_range = st.slider(f"Select Range ({poll_col}):", min_poll, max_poll, (min_poll, max_poll), step=0.1)
                       else:
                            st.caption(f"Only one value available for {poll_col}: {min_poll:.2f}")
                            selected_poll_range = (min_poll, max_poll)
                  else:
                       selected_poll_range = None
                       st.caption(f"No valid numeric data in '{poll_col}'.")
             else:
                  selected_poll_range = None
                  st.caption(f"'{poll_col}' column not found.")

        df_filtered = df_original_processed.copy()
        try:
            if selected_countries and 'Country' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
            if selected_year_range and 'Start_Year' in df_filtered.columns:
                df_filtered['Start_Year'] = pd.to_numeric(df_filtered['Start_Year'], errors='coerce')
                df_filtered = df_filtered[
                    (df_filtered['Start_Year'] >= selected_year_range[0]) &
                    (df_filtered['Start_Year'] <= selected_year_range[1])
                ]
            if selected_poll_range and poll_col in df_filtered.columns:
                df_filtered[poll_col] = pd.to_numeric(df_filtered[poll_col], errors='coerce')
                df_filtered = df_filtered[
                    (df_filtered[poll_col] >= selected_poll_range[0]) &
                    (df_filtered[poll_col] <= selected_poll_range[1])
                ]
        except Exception as e:
            st.error(f"Error applying filters: {e}")
            df_filtered = df_original_processed.copy()

        st.subheader("Sort Data")
        sort_column = st.selectbox("Sort by column:", options=[''] + sorted(df_filtered.columns.tolist()))
        sort_ascending = st.radio("Sort order:", ('Ascending', 'Descending')) == 'Ascending'

        if sort_column:
             try:
                 if pd.api.types.is_numeric_dtype(df_filtered[sort_column].dtype):
                     df_display = df_filtered.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')
                 else:
                      df_display = df_filtered.sort_values(by=sort_column, ascending=sort_ascending, na_position='last')
             except Exception as e:
                 st.error(f"Error sorting by {sort_column}: {e}")
                 df_display = df_filtered
        else:
            df_display = df_filtered

        st.subheader("Filtered and Sorted Data")
        st.dataframe(df_display)

        st.metric("Number of Rows Displayed", len(df_display))

        st.subheader("Quick Visualizations")
        plot_cols = df_display.select_dtypes(include=np.number).columns.tolist()
        if len(plot_cols) >= 2:
            col_vis1, col_vis2 = st.columns(2)
            with col_vis1:
                x_axis = st.selectbox("Select X-axis for Scatter Plot:", options=plot_cols, index=0)
            with col_vis2:
                 y_axis = st.selectbox("Select Y-axis for Scatter Plot:", options=plot_cols, index = 1 if len(plot_cols) > 1 else 0)

            if x_axis and y_axis:
                 try:
                    fig_scatter, ax_scatter = plt.subplots()
                    ax_scatter.scatter(df_display[x_axis], df_display[y_axis], alpha=0.3, s=10)
                    ax_scatter.set_xlabel(x_axis)
                    ax_scatter.set_ylabel(y_axis)
                    ax_scatter.set_title(f"Scatter Plot: {y_axis} vs {x_axis}")
                    ax_scatter.grid(True, linestyle=':', alpha=0.6)
                    st.pyplot(fig_scatter)
                 except Exception as e:
                     st.error(f"Error creating scatter plot: {e}")
            else:
                 st.caption("Select valid X and Y axes.")
        else:
            st.caption("Not enough numerical columns for scatter plot.")

    else:
        st.warning(f"Original processed dataset (`{ORIGINAL_DATA_PATH}`) could not be loaded. Cannot display explorer.")