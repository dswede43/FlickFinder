# FlickFinder
My machine learning movie recommendation system!

## Dataset
The TMDB movies dataset from Kaggle (https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) was used to train the model.

## ML model
The model was created by converting the text of movie overviews into Term Frequency-Inverse Document Frequencies (TF-IDF), followed by pairwise calculations of cosine similarity between all movies.

## Dependencies
Two data files are required to run the app:

1. A subset of the movies data frame found [here](https://files.stratz.me/FlickFinder/movies.parquet)
2. The model matrix found [here](https://files.stratz.me/FlickFinder/IF-IDF_cosine_sim_model.parquet)

## Docker
A docker image of this app can also be pulled from my Docker Hub [repository](https://hub.docker.com/r/dswede43/flickfinder).
