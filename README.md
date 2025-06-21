
# Water Quality Clustering Analysis Dashboard

Welcome to the interactive dashboard for exploring water quality clustering results! This application presents findings from in-depth clustering analysis, focusing on the performance and interpretation of different clustering algorithms after hyperparameter tuning.

## Project Overview

This project analyzes water quality data to uncover natural groupings within water samples based on various measured features. Using dimensionality reduction (Principal Component Analysis, PCA), the data is prepared for clustering with several unsupervised algorithms. The results are visualized and explored through a user-friendly dashboard.

**Main Goals:**
- Identify meaningful clusters in water quality data.
- Compare clustering algorithms and their performance after hyperparameter optimization.
- Provide interactive visualizations and profiling for deeper understanding of the clusters.

## Dashboard Features

Navigate using the sidebar to explore:

- **Model Performance:**  
  Compare evaluation metrics for optimized clustering models, such as Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score, and the number of clusters found.

- **Cluster Visualization:**  
  Visualize clusters projected onto 2D PCA space for intuitive understanding of group separations.

- **Cluster Profiling:**  
  Analyze the characteristics of each cluster using radar charts and country-wise distributions, providing insights into what defines each group.

- **Dataset Explorer:**  
  Interactively filter and view the original processed dataset (shape: 18,135 rows × 28 columns), with tools for sorting, filtering by country, year, and feature ranges, and generating scatter plots for quick insights.

## Clustering Workflow

1. **Data Preprocessing:**  
   The water quality dataset is cleaned and dimensionality reduction is performed using PCA.

2. **Clustering Algorithms:**  
   Multiple clustering algorithms (e.g., KMeans, AgglomerativeClustering, DBSCAN) are applied with various hyperparameter configurations.

3. **Model Evaluation:**  
   Performance is evaluated using metrics such as Silhouette Score (higher is better), Calinski-Harabasz Score (higher is better), and Davies-Bouldin Score (lower is better). Models are ranked based on these metrics.

4. **Visualization & Profiling:**  
   - Best performing models (e.g., KMeans: Silhouette Score 0.908) are highlighted.
   - Users can visualize clusters, explore their profiles, and examine country distributions per cluster.

## Example Results

- **Best Performing Model:**  
  - KMeans (Silhouette Score: 0.908)

- **Cluster Profiling:**  
  - Clusters are profiled on key features and distributions visualized for interpretation.
  - Example: AgglomerativeClustering assigned valid cluster labels to 18,135 rows, with profiles per cluster.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BryanTheLai/MLAssignment-ClusteringV2.git
   ```
2. **Install dependencies:**  
   (e.g., via `requirements.txt` or manually: `pip install pandas numpy scikit-learn matplotlib seaborn dash`)
3. **Run the dashboard:**
   ```bash
   cd dashboard
   python app.py
   ```
4. **Access the dashboard:**  
   Open your browser to `http://localhost:8050`.

## Data

- **Source:**  
  Processed water quality dataset located at `MachineLearningAssignment/data/processed/processed_waterPollution.csv`
- **Shape:**  
  18,135 rows × 28 columns

## Technologies Used

- Jupyter Notebook, Python
- scikit-learn, pandas, numpy, matplotlib, seaborn
- dash (for interactive dashboard)

## Contributing

Contributions and suggestions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

---

Let me know if you want this README further tailored or if you need sections on setup, requirements, or code structure!
