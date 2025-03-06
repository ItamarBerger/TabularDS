import numpy as np

# """
# Convert movieLens 20M dataset into transactional dataset
# """
# # load 
import pandas as pd

# data = pd.read_csv('final_project/data/rating.csv')
# movies_data = pd.read_csv('final_project/data/movie.csv')

# movie_popularity = data.groupby('movieId')['userId'].nunique().sort_values(ascending=False)

# # Select the top 10% most popular movies
# top_10_percent_count = int(0.15 * len(movie_popularity))
# print(f"Top 10% count: {top_10_percent_count}")
# top_movies = movie_popularity.head(top_10_percent_count).index.tolist()

# # Filter dataset to only include these movies
# filtered_df = data[data['movieId'].isin(top_movies)]


# user_movie_matrix = filtered_df.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='count').fillna(0)
# user_movie_matrix = user_movie_matrix.map(lambda x: 1 if x > 0 else 0)  # Convert to binary


# # preprocess movies with titles
# user_movie_matrix_with_titles = user_movie_matrix.copy()
# movie_titles = movies_data.set_index('movieId')['title']
# user_movie_matrix_with_titles.columns = movie_titles[user_movie_matrix_with_titles.columns]

# user_movie_matrix_with_titles.to_csv('final_project//data/movies_with_clusters.csv', index=False)



# """
# Convert Online Retails dataset into transactional
# """
# import pandas as pd
from collections import deque

# # Sample data
# data = pd.read_csv('final_project/data/online_retail.csv')
# # Create DataFrame with relevant columns
# df = pd.DataFrame(data, columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'])

# # Get the number of unique products purchased by each customer, sorted in descending order
# product_popularity = df.groupby('Description')['CustomerID'].nunique().sort_values(ascending=False)


# # Top 15% of customers based on product count
# top_10_percent_count = int(0.15 * len(product_popularity))


# # Get the products purchased by the top 15% of customers
# top_prods = product_popularity.head(top_10_percent_count).index.tolist()

# print(top_prods)


# # Filter the original dataset for the products purchased by the top customers
# filtered_df = df[df['Description'].isin(top_prods)]

# # Create a transactional matrix where each cell indicates whether a customer bought the product
# transaction_matrix = filtered_df.pivot_table(
#     index='CustomerID', 
#     columns='Description', 
#     values='Quantity', 
#     aggfunc=lambda x: 1,  # Indicate presence (purchase) of the item
#     fill_value=0  # Fill NaNs with 0 (i.e., no purchase)
# )

# # Save the transactional matrix to a CSV file with CustomerID included
# transaction_matrix.to_csv('final_project/data/online_retail_transactional.csv', index=False)




"""
Convert Music datdaset to transactional CSV file
"""


# import pandas as pd


# df_artists = pd.read_csv("final_project/data/artists.dat", delimiter="\t")

# # Read the .dat file (tab-separated)
# df = pd.read_csv("final_project/data/user_artists.dat", delimiter="\t")

# # Rename columns for clarity
# df.columns = ["userID", "artistID", "weight"]

# # Find the top 10% most popular artists
# top_1_percent_threshold = int(len(df["artistID"].unique()) * 0.015)
# top_artists = df["artistID"].value_counts().nlargest(top_1_percent_threshold).index

# #  Filter dataset to include only the top 1% artists
# df_filtered = df[df["artistID"].isin(top_artists)]

# #  Convert to transactional format (pivot table)
# df_transactional = df_filtered.pivot_table(index="userID", columns="artistID", values="weight", aggfunc="count", fill_value=0)

# # Convert to binary (1 = listened, 0 = not listened)
# df_transactional = (df_transactional > 0).astype(int)


# for col in df_transactional.columns:
#     df_transactional.rename(columns={col: df_artists[df_artists['id'] == col].iloc[0]['name']}, inplace=True)

# # Display the first few rows
# print(df_transactional.head())



# # Save as CSV (comma-separated)
# df_transactional.to_csv("final_project/data/user_artists_transactional.csv", index=False)



"""
Convert Netflix Prize dataset into transactional - preprocess inspired by kaggle notebook: https://www.kaggle.com/code/morrisb/how-to-recommend-anything-deep-recommender
"""

# Load single data-file
import pandas as pd
from collections import deque

# Load the dataset
df_raw = pd.read_csv('final_project/data/combined_data_1.txt', header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

# Find empty rows to slice the dataframe for each movie
tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

# Shift the movie_indices by one to get start and endpoints of all movies
shifted_movie_indices = deque(movie_indices)
shifted_movie_indices.rotate(-1)

# Gather all dataframes
user_data = []

# Iterate over all movies
for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):
    # Check if it is the last movie in the file
    if df_id_1 < df_id_2:
        tmp_df = df_raw.loc[df_id_1+1:df_id_2-1].copy()
    else:
        tmp_df = df_raw.loc[df_id_1+1:].copy()

    # Create movie_id column
    tmp_df['Movie'] = movie_id
    
    # Append dataframe to list
    user_data.append(tmp_df)

# Combine all dataframes
df = pd.concat(user_data)

# Calculate the frequency of ratings per movie and determine the top 20%
movie_ratings_count = df['Movie'].value_counts()
top_20_percent_movies = movie_ratings_count.head(int(len(movie_ratings_count))).index

# Filter the dataframe to only include the top 20% movies
df_filtered = df[df['Movie'].isin(top_20_percent_movies)]

# Convert to transactional format: rows are users, columns are top 20% movies
transactional_data = df_filtered[0:50000]
transactional_data = pd.pivot_table(transactional_data, index='User', columns='Movie', aggfunc=lambda x: 1, fill_value=0)

# Load movie titles
movie_titles_df = pd.read_csv('final_project/data/movie_titles.csv', encoding='ISO-8859-1', on_bad_lines='skip', header=None, names=['id', 'year', 'title'])
print(movie_titles_df.head())

# Rename columns with movie titles
# Rename columns with movie titles
# Now, the movie_ids are in the 'Movie' column in transactional_data, so we need to map movie_id to the title
movie_titles_dict = movie_titles_df.set_index('id')['title'].to_dict()

# Replace the movie IDs in the columns with titles
transactional_data.rename(columns=movie_titles_dict, inplace=True)

# Display the transactional dataset
print(transactional_data.head())

# Save the transactional dataset
transactional_data.to_csv('final_project/data/netflix_transactional.csv', index=False)
