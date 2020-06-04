#MOVIE RECOMMENDATION SYSTEM USING RNN

#IMPORTING ALL THE LIBRARIES
import pandas as pd
import numpy as np

#IMPORTING DATASET
#usecol is used to lod specific columns from the dataset
movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
movies_df.head()
rating_df.head()

#MERGING THE ABOVE TWO DATASETS BASED ON THE SAME COLUMN
df=pd.merge(rating_df,movies_df,on='movieId')
df.head()

#COUNTING NUMBER OF RATING EACH MOVIE HAS GOT
combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head()

#MERGING TOTALRATINGCOUNT COLUMN BACK TO OUR DATASET
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
rating_with_totalRatingCount.head()

#PRINTING MOVIE WHO HAS POPULARITY GREATER THAN THRESHOLD
popularity_threshold = 50
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_movie.head()

#CREATING PIVOT MATRIX
movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
movie_features_df.head()

#CONVERTING TABLE TO ARRAY MARTIX
from scipy.sparse import csr_matrix
movie_features_df_matrix = csr_matrix(movie_features_df.values)

#IMPORTING NN(we use cosine matrix for finding the nearest neighbor)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)
#P IS EUCLEDIAN DISTANCE PARAMETER

#MALING A RANDOM CHOICE
query_index = np.random.choice(movie_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

#PREDICTING TOP 5 MOVIESA ACCORDING TO THE RANDOM CHOICE
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
