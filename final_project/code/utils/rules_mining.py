import numpy as np

def print_basic_metrics(rules):
  # Calculate the average support, confidence, and lift from the rules DataFrame
  avg_support = np.mean(rules['support'])
  avg_confidence = np.mean(rules['confidence'])
  avg_lift = np.mean(rules['lift'])

  # Print the metrics
  print(f"\nMetrics: Support average: {avg_support:.4f}, Confidence average: {avg_confidence:.4f}, Lift average: {avg_lift:.4f}")



import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules

def plot_topk_rules_in_graph(rules, frequent_itemsets, k=10):

    # Create a directed graph
    G = nx.DiGraph()
    narrow_rules = rules.sort_values(by='lift', ascending=False).head(k)
    # Add nodes and directed edges
    for _, rule in narrow_rules.iterrows():
        antecedent = ', '.join(list(rule['antecedents']))
        consequent = ', '.join(list(rule['consequents']))

        # Add a directed edge from antecedent to consequent
        G.add_edge(antecedent, consequent)

    # Draw the directed graph
    plt.figure(figsize=(18, 13))
    pos = nx.spring_layout(G, k=0.5, iterations=k)  # Layout for the graph

    # Draw the nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowstyle='->', arrowsize=3)
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black')

    # Add edge labels (optional, if you want to display lift, confidence, or support)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Sample Association Rules Directed Graph")
    plt.axis('off')
    plt.show()





import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics_boxplots(rules):
  if not rules.empty:


      print(f"max lift value is {rules['lift'].max():.4f}")
      print(f"max confidence value is {rules['confidence'].max():.4f}")

      fig, axes = plt.subplots(1, 2, figsize=(8, 5))

      # plot Confidence
      sns.boxplot(data=rules['confidence'], ax=axes[0])
      axes[0].set_title('Confidence')
      axes[0].set_ylabel('Confidence')

      # plot Lift
      sns.boxplot(data=rules['lift'], ax=axes[1])
      axes[1].set_title('Lift')
      axes[1].set_ylabel('Lift')

      # show plot
      plt.tight_layout()
      plt.show()
  else:
      print("No rules to plot.")



      # Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in divide*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*DataFrames with non-bool types result in worse computationalperformance*")

from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np

# Function to compute global support
import numpy as np

def calculate_global_support(itemset, full_data):
    """Calculate global support of an itemset across the entire transactions dataset efficiently."""
    itemset = list(itemset)  # Ensure itemset is a list
    relevant_rows = full_data[itemset].sum(axis=1) == len(itemset)  # Check if all items are present
    return relevant_rows.mean()
