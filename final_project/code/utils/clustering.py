

from sklearn.cluster import KMeans

def add_cluster_users_col(num_clusters, transactions_matrix):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    user_clusters = kmeans.fit_predict(transactions_matrix)
    transactions_matrix['Cluster'] = user_clusters



def  filter_valid_clusters_by_size(df):
  # Calculate the percentage of each cluster
  cluster_percentage = df['Cluster'].value_counts(normalize=True) * 100

  # Get the clusters with percentage >= 1
  valid_clusters = cluster_percentage[(cluster_percentage >= 1) & (cluster_percentage <= 30)].index

  # Filter the dataframe to include only rows from valid clusters
  df_filtered = df[df['Cluster'].isin(valid_clusters)]
  return df_filtered



import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_distribution(df):
  # Get the distribution of clusters (as percentages only)
  cluster_percentages = (df['Cluster'].value_counts(normalize=True) * 100).sort_index()

  # Plot the distribution using seaborn barplot
  plt.figure(figsize=(10, 6))
  sns.barplot(x=cluster_percentages.index, y=cluster_percentages.values, palette='viridis')

  # Add labels and title
  plt.title('Distribution of Items Across Clusters (Percentage)', fontsize=16)
  plt.xlabel('Cluster', fontsize=12)
  plt.ylabel('Percentage of Items (%)', fontsize=12)

  # Remove x-tick labels (to avoid showing the cluster index)
  plt.xticks([])

  # Show the plot
  plt.tight_layout()
  plt.show()



  def calculate_sparsity(cluster_data):
    """Calculate the sparsity of the cluster data."""
    total_entries = cluster_data.size
    non_zero_entries = cluster_data.values.sum()
    return 1 - (non_zero_entries / total_entries)

def plot_sparsity(cluster_data, cluster_id):
    """Plot the sparsity of a cluster."""
    sparsity = calculate_sparsity(cluster_data)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cluster_data, cbar_kws={'label': 'Binary Value'}, cmap='Blues')
    plt.title(f"Sparsity Heatmap for Cluster {cluster_id} (Sparsity: {sparsity:.2f})")
    plt.show()

def plot_item_frequency_distribution(cluster_data, cluster_id):
    """Plot item frequency distribution for a cluster."""
    item_frequencies = cluster_data.sum(axis=0)  # Sum across rows to get item frequency
    plt.figure(figsize=(10, 6))
    sns.histplot(item_frequencies, bins=30, kde=True)
    plt.title(f"Item Frequency Distribution for Cluster {cluster_id}")
    plt.xlabel('Item Frequency (Number of Users)')
    plt.ylabel('Frequency')
    plt.show()
