from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import path

np.random.seed(13)
pd.set_option('display.max_columns', None)
# all columns in one line
pd.set_option('expand_frame_repr', False)

csvPath = "C:\\Users\\esinagra\\Desktop\\Nuova cartella\\Magistrale\\2\\DATA ANALYTICS\\progetto\\ml-25m"
files = ["movies.csv", "ratings.csv", "genome-scores.csv", "genome-tags.csv", "links.csv", "tags.csv"]

df_movies = pd.read_csv(path.join(csvPath, files[0]))
df_ratings = pd.read_csv(path.join(csvPath, files[1]))
df_genome_scores = pd.read_csv(path.join(csvPath, files[2]))
df_genome_tags = pd.read_csv(path.join(csvPath, files[3]))
#df_links = pd.read_csv(path.join(csvPath, files[4]))
#df_tags = pd.read_csv(path.join(csvPath, files[5]))

#df = df_movies['genres'].str.split('|', expand=True)
mlb = MultiLabelBinarizer(sparse_output=True)
df_movies = df_movies.join(pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df_movies.pop('genres').str.split('|')),
                index=df_movies.index,
                columns=mlb.classes_))
print(df_movies.head())

df_genome = pd.merge(df_genome_tags, df_genome_scores, on="tagId")
df_genome = df_genome.pivot(index='movieId', columns='tag', values="relevance")
df = pd.merge(df_movies, df_genome, on="movieId")
df_ratings = df_ratings.iloc[:,[1,2]]
print(df_ratings)

df_ratings = df_ratings.groupby(by='movieId').mean()
df = pd.merge(df_ratings, df, on="movieId")
print(df.info())

# SPLIT: TRAIN, VALIDATION, TEST
df_X = df.loc[:,df.columns != 'rating']
df_Y = df['rating']

# PRE DATA VISUALIZATION
df_Y.plot(kind ='density')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=20)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=20)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)
print(X_train.head())

# NORMALIZATION
Y_train = (Y_train - Y_train.mean()) / Y_train.std()
Y_train.plot(kind ='density')
plt.show()