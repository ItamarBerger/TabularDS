# TabularDS Course



# Final Project: Improving Association Rule Mining using Clustering-Based Item Similarity

This repository contains the code, data, results, and visualizations for my final project. The main goal of the project is to apply and compare a proposed solution for ARM with a baseline approach using 4 datasets. Below is an overview of the folder structure and where to find what in the repository.

## Folder Structure

### 1. `final_project/`
The root folder containing all the main components of the project.

### 2. `code/`
This folder contains all the Python code and notebooks related to the project.

- **`example_notebook.ipynb`**: This Jupyter notebook demonstrates the solution, comparing the baseline approach with the proposed solution. It walks you through the steps and provides insights into the results.

- **`utils/`**: This sub-folder contains utility scripts used for preprocessing and applying the proposed methods:
  - **`utils.py`**: Includes preprocessing steps for all four datasets with detailed comments and explanations, including the conversion to transactional datasets.
  - **`clustering.py`**: Contains the code for clustering the datasets using the proposed approach.
  - **`rules_mining.py`**: Contains the code for applying rule mining and plotting the results.
  - **`requirements.txt`**: Contains the non-native python libraries included in the example notebook.

### 3. `data/`
This folder contains the raw datasets used in the project. It's mostly for internal use, as the preprocessing steps are already included in the code.

### 4. `results/`
This folder contains the results of applying the methods to each dataset.


- **Jupyter Notebooks for Each Dataset**: For each of the four datasets, there is a corresponding Jupyter notebook outlining the results and visualizations. These notebooks demonstrate how the proposed method performs compared to the baseline approach, including any plots and analysis.

### 5. `Visualization/`
This folder contains visualizations primarily for my personal use, including any figures or plots that were generated during the process. These are all provided in the relevant notebooks mentioned above.


### 6. **report.pdf**
This file contains a full report describing the background, solution overview, results and related work relevant to the project.

