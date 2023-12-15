# Requiered imports

from spotlight.datasets.movielens import get_movielens_dataset
import pandas as pd

def get_merge_data():
    """
    Merge data from Spotlight with MovieLens dataset to obtain movie genres.

    This function performs the following steps:
    1. Downloads the MovieLens dataset using the Spotlight library.
    2. Extracts item IDs and ratings from the downloaded dataset.
    3. Downloads the MovieLens dataset containing movie genres.
    4. Renames the 'movieId' column to 'item_ids' in the genres dataset.
    5. Merges the two datasets using the 'item_ids' column as the key.

    Returns:
        df (pd.DataFrame): A DataFrame containing merged data with movie genres.
    """
    # Download the dataset movielens from spotlight
    dataset = get_movielens_dataset(variant='20M')
    item_ids = dataset.item_ids
    ratings = dataset.ratings
    df = pd.DataFrame({'item_ids':item_ids,'ratings':ratings})

    # Download movielens dataset to add genres
    genres_df = pd.read_csv('../Data/movies.csv')

    # Merge dataframes by item_ids
    df = pd.merge(df,genres_df,on='item_ids')
    return df

class TopPopRecommender:
    '''
    Top Popular Recommender model. This model will be used in the case that the user only takes as input genre, without
    any film suggestion. In that case, a popular recommender is implemented, which will recommend the most popular films
    on that genre.
    '''
    def fit(self, train):
        '''
        Fit the TopPopRecommender model to the training data.

        Parameters:
        train (DataFrame): A pandas DataFrame containing the training data with columns 'item_ids' and 'genres'.
        '''
        self.genre_popular_items = {}
        genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                  'Western']

        for genre in genres:
            # Filter the train dataset by the specified genre
            genre_items = train[train['genres'].str.contains(genre)]
            # Group by 'item_ids' and count ratings
            genre_item_popularity = genre_items[['item_ids', 'ratings']].groupby(by='item_ids').count()
            # Sort by popularity
            genre_popular_items = genre_item_popularity.sort_values(by='ratings', ascending=False).index
            self.genre_popular_items[genre] = genre_popular_items

    def predict(self, at=5, genre=None):
        '''
        Generate movie recommendations using the TopPopRecommender model.

        Parameters:
        at (int): The number of recommendations to generate.
        genre (str): The genre for which recommendations are desired.

        Returns:
        list: A list of recommended movie IDs.
        '''
        if not hasattr(self, 'genre_popular_items'):
            raise ValueError("The model has not been trained. Call 'fit' first.")

        if genre is not None:
            if genre not in self.genre_popular_items:
                raise ValueError(f"Genre '{genre}' not found in training data.")
            recommended_items = self.genre_popular_items[genre][0:at]
        else:
            raise ValueError("You must specify a genre for prediction.")

        return recommended_items

def from_id_to_title(recommended_items, df):
    '''
    This function will return the title of the recommended items.
    Parameters:
    recomendad_items (list): list of recommended items
    df (pd.DataFrame): DataFrame with the movies dataset
    Returns:
    list: A list of recommended movie titles.
    '''
    result_df = df[df['item_ids'].isin(recommended_items)]
    result_df = result_df.sort_values(by=['item_ids'], key=lambda x: x.map({v: i for i, v in enumerate(recommended_items)}))
    return list(result_df['title'].unique())

