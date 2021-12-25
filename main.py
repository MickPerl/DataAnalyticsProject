from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from os import path

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
print(df.head())