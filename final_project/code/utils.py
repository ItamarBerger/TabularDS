import numpy as np

"""
Convert movieLens 20M dataset into transactional dataset
"""
# # load 
# import pandas as pd

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



"""
Convert Online Retails dataset into transactional
"""
import pandas as pd

# Sample data
data = pd.read_csv('final_project/data/online_retail.csv')
# Create DataFrame with relevant columns
df = pd.DataFrame(data, columns=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'])

# Get the number of unique products purchased by each customer, sorted in descending order
product_popularity = df.groupby('Description')['CustomerID'].nunique().sort_values(ascending=False)


# Top 15% of customers based on product count
top_10_percent_count = int(0.15 * len(product_popularity))


# Get the products purchased by the top 15% of customers
top_prods = product_popularity.head(top_10_percent_count).index.tolist()

print(top_prods)


# Filter the original dataset for the products purchased by the top customers
filtered_df = df[df['Description'].isin(top_prods)]

# Create a transactional matrix where each cell indicates whether a customer bought the product
transaction_matrix = filtered_df.pivot_table(
    index='CustomerID', 
    columns='Description', 
    values='Quantity', 
    aggfunc=lambda x: 1,  # Indicate presence (purchase) of the item
    fill_value=0  # Fill NaNs with 0 (i.e., no purchase)
)

# Save the transactional matrix to a CSV file with CustomerID included
transaction_matrix.to_csv('final_project/data/online_retail_transactional.csv', index=True)
