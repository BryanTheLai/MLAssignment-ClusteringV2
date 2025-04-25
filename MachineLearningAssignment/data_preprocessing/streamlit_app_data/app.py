# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# --- Page Configuration (Optional but Recommended) ---
st.set_page_config(
    page_title="Water Body Clustering Analysis",
    page_icon="🌊",
    layout="wide"
)

# --- Constants ---
DATA_DIR = 'streamlit_app_data' # Directory where Part B outputs are saved
PCA_DATA_FILE = os.path.join(DATA_DIR, 'pca_2D_data.csv') # Assuming 2D focus
PARAMS_FILE = os.path.join(DATA_DIR, 'optimized_model_params.joblib')
SCORES_FILE = os.path.join(DATA_DIR, 'optimized_model_scores.csv')
PCA_OBJECT_FILE = os.path.join(DATA_DIR, 'pca_2D_object.joblib') # For axis labels

# --- Load Data ---
# Use Streamlit caching for efficiency
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data():
    """Loads all necessary data artifacts."""
    try:
        pca_df = pd.read_csv(PCA_DATA_FILE)
        params_all = joblib.load(PARAMS_FILE)
        scores_df = pd.read_csv(SCORES_FILE)
        pca_obj = joblib.load(PCA_OBJECT_FILE)

        # Load labels for each model (handle potential missing files)
        labels_all = {}
        model_names = params_all.keys() # Get model names from the params keys
        for model_name in model_names:
            # Construct expected label filename (adjust if your naming differs)
            label_filename = f'best_labels_{model_name}_2D.npy'
            label_filepath = os.path.join(DATA_DIR, label_filename)
            if os.path.exists(label_filepath):
                labels_all[model_name] = np.load(label_filepath)
            else:
                st.warning(f"Label file not found for {model_name}: {label_filepath}")
                labels_all[model_name] = None # Mark as unavailable

        return pca_df, params_all, scores_df, labels_all, pca_obj, list(model_names)

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Make sure all necessary files are in the '{DATA_DIR}' folder.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None, None, None

# Load the data
pca_df, params_all, scores_df, labels_all, pca_obj, available_models = load_data()

# --- Main App Structure ---
st.title("🌊 Water Body Clustering Optimization Results")
st.markdown("Interactive dashboard showing the results of hyperparameter tuning for various clustering models on 2D PCA data.")

# Check if data loaded successfully
if pca_df is not None and params_all is not None and scores_df is not None and labels_all is not None:

    # --- Sidebar for Model Selection ---
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose an optimized model to view:",
        options=available_models,
        index=0 # Default to the first model in the list
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"Displaying results for: **{selected_model_name}**")

    # --- Display Results for Selected Model ---
    st.header(f"Results for: {selected_model_name}")

    # Get data for the selected model
    model_params = params_all.get(selected_model_name, {})
    # Find the scores row matching the selected model name
    model_scores = scores_df[scores_df['Algorithm'] == selected_model_name]
    model_labels = labels_all.get(selected_model_name)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Optimized Parameters")
        if model_params:
            st.json(model_params) # Pretty print dictionary
        else:
            st.write("Parameters not available.")

        st.subheader("Performance Metrics")
        if not model_scores.empty:
            score_row = model_scores.iloc[0]
            st.metric("Silhouette Score", f"{score_row['Silhouette']:.4f}" if pd.notna(score_row['Silhouette']) else "N/A")
            st.metric("Davies-Bouldin", f"{score_row['Davies-Bouldin']:.4f}" if pd.notna(score_row['Davies-Bouldin']) else "N/A")
            st.metric("Calinski-Harabasz", f"{score_row['Calinski-Harabasz']:.0f}" if pd.notna(score_row['Calinski-Harabasz']) else "N/A")
            st.metric("Number of Clusters", f"{int(score_row['Num_Clusters'])}" if pd.notna(score_row['Num_Clusters']) else "N/A")
        else:
             st.write("Scores not available.")

    with col2:
        st.subheader("Cluster Visualization (2D PCA)")
        if model_labels is not None and pca_obj is not None:
            # Create a copy for plotting to avoid modifying cached data
            plot_df = pca_df.copy()
            plot_df['Cluster'] = model_labels
            # Convert cluster labels to string for categorical coloring in Plotly
            # Handle noise points (-1) explicitly
            plot_df['Cluster'] = plot_df['Cluster'].astype(str)
            plot_df['Cluster'] = plot_df['Cluster'].replace('-1', 'Noise') # Label noise clearly

            # Define axis labels using the loaded PCA object
            pc1_var = pca_obj.explained_variance_ratio_[0] * 100
            pc2_var = pca_obj.explained_variance_ratio_[1] * 100
            x_label = f"Principal Component 1 ({pc1_var:.2f}% Variance)"
            y_label = f"Principal Component 2 ({pc2_var:.2f}% Variance)"

            # Create interactive plot
            fig = px.scatter(
                plot_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title=f'Optimized Clusters ({selected_model_name})',
                labels={'PC1': x_label, 'PC2': y_label}, # Use formatted labels
                category_orders={"Cluster": sorted(plot_df['Cluster'].unique(), key=lambda x: int(x) if x != 'Noise' else -1)}, # Sort clusters numerically
                color_discrete_map={'Noise': 'grey'} # Make noise points grey
            )

            fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
            fig.update_layout(legend_title_text='Cluster Label')
            st.plotly_chart(fig, use_container_width=True)

        elif pca_obj is None:
             st.warning("PCA object not loaded. Cannot display axis variance.")
             # Add basic plot without variance if needed
        else:
            st.warning(f"Cluster labels are not available for {selected_model_name}. Cannot display plot.")

else:
    st.error("Data loading failed. Cannot display the application.")


st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Water Body Pollution Clustering Assignment.")