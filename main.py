import pandas as pd
from os import path

pd.set_option('display.max_columns', None)

csvPath = "C:\\Users\\esinagra\\Desktop\\Nuova cartella\\Magistrale\\2\\DATA ANALYTICS\\progetto\\ml-25m"
files = ["movies.csv", "ratings.csv", "genome-scores.csv", "genome-tags.csv", "links.csv", "tags.csv"]

df_movies = pd.read_csv(path.join(csvPath, files[0]))
df_ratings = pd.read_csv(path.join(csvPath, files[1]))
df_genome_scores = pd.read_csv(path.join(csvPath, files[2]))
df_genome_tags = pd.read_csv(path.join(csvPath, files[3]))
#df_links = pd.read_csv(path.join(csvPath, files[4]))
#df_tags = pd.read_csv(path.join(csvPath, files[5]))

print(df_movies.head())
print(df_genome_tags.head())
#print(df_genome_scores.head())
#df = df_movies.join(df_genome_scores, on="movieId")
df = pd.concat([df_movies, df_genome_tags['tag']])
print(df.head())
#df = pd.merge(df_movies, df_genome_scores, on="movieId")
#print(df.head())
quit(0)