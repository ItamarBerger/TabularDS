import numpy as np

"""
Convert movieLens 20M dataset into transactional dataset
"""
# load 
import pandas as pd

data = pd.read_csv('final_project/data/rating.csv')
movies_data = pd.read_csv('final_project/data/movie.csv')

movie_popularity = data.groupby('movieId')['userId'].nunique().sort_values(ascending=False)

# Select the top 10% most popular movies
top_10_percent_count = int(0.01 * len(movie_popularity))
top_movies = movie_popularity.head(top_10_percent_count).index.tolist()

# Filter dataset to only include these movies
filtered_df = data[data['movieId'].isin(top_movies)]


user_movie_matrix = filtered_df.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='count').fillna(0)
user_movie_matrix = user_movie_matrix.map(lambda x: 1 if x > 0 else 0)  # Convert to binary


# preprocess movies with titles
user_movie_matrix_with_titles = user_movie_matrix.copy()
movie_titles = movies_data.set_index('movieId')['title']
user_movie_matrix_with_titles.columns = movie_titles[user_movie_matrix_with_titles.columns]

user_movie_matrix_with_titles.to_csv('final_project//data/movies_with_clusters.csv', index=False)



"""
Convert Online Retails dataset into transactional
"""