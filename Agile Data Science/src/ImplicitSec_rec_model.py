from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.interactions import Interactions
from spotlight.sequence.representations import CNNNet
from spotlight.evaluation import sequence_mrr_score
import numpy as np
import torch  # to save the model. Backbone done in torch.
import pandas as pd


def load_data_to_sequences(variant = '20M', 
                            max_sequence_length = 200,
                            min_sequence_length = 20,
                            step_size = 200, df_split = False, 
                            retrain = False, username = None, 
                             add_feedback_to_retrain = None):
    
    """This functions loads the movielens dataset from the spotlight package
    
    Parameters: 
        variant (str) : Parameter specifying the desired variant of the movielens
                        dataset. '20M' by default. 
                        Possible variant inputs: '100K','1M','10M','20M'
        max_sequence_length (int) = 200 by default.
        min_sequence_length (int) = 20 by default.
        step_size (int) = 200 by default.
        df_split (boolean): False by default. Determines whether the train/test split is conducted
    Returns:
        train (sequence)
        test (sequence) 
        
    """
    random_state = np.random.RandomState(100)
    dataset = get_movielens_dataset(variant=variant)
    if retrain: 
        dataset = add_feedback_to_retrain(username)
    if df_split:
        train, test = user_based_train_test_split(dataset,
                                                random_state=random_state)
        train = train.to_sequence(max_sequence_length=max_sequence_length,
                                min_sequence_length=min_sequence_length,
                                step_size=step_size)
        test = test.to_sequence(max_sequence_length=max_sequence_length,
                                min_sequence_length=min_sequence_length,
                                step_size=step_size)
        
        return train, test
    
    else:
        dataset = dataset.to_sequence(max_sequence_length=max_sequence_length,
                                min_sequence_length=min_sequence_length,
                                step_size=step_size) 
        return dataset


def train_ImplicitSec_model(train, model_type = 'cnn', save_model = True, filename = 'ImplicitSec_rec_model'):
    """Function that trains and saves the recommender model ImplicitSequenceModel() 

    Args:
        train (sequence): sequence data ready to train using the ImplicitSequenceModel() 
        model_type (str, optional): Sequence representation to use by the ImplicitSequenceModel.
                                    Default to 'cnn'. 
                                    Possible inputs: 'cnn', 'pooling' or 'lstm'
        save_model (boolean): Default to 'True', determins whether or not to save the model. 

    Returns:
        model (spotlight.sequence.implicit.ImplicitSequenceModel): trained model
    """
    model = ImplicitSequenceModel(n_iter=3,
                                  representation=model_type,
                                  loss='bpr')
    model.fit(train,verbose=True)
    if save_model: 
        torch.save(model, '../trained_models/'+filename+'.pth')
    
    return model 

def evaluate_model(test, model): 
    """evaluates the trained model by computing the mrr metrics

    Args:
        test (sequence): _description_
        model (spotlight.sequence.implicit.ImplicitSequenceModel): _description_

    Returns:
        mrr (np.array): mrr metrics
    """
    mrr = sequence_mrr_score(model, test)
    print('MRR: ', mrr)
    
    return mrr

def predict(model, input_movie_ids, genres_df, genre = None,  at = 5):
    """Given an specific array of input movies, 
        this function returns a prediction of the top 5 movies rated 
        as a result of the item-to-item recommender system
        If a genre is provided, the output is filtered and only provides
        movies from that genre. 

    Args:
        model (spotlight.sequence.implicit.ImplicitSequenceModel): already trained model
        input_movie_ids (np.array): list of input movies from which the top 5 movies are
                                    recommended
        genres_df (pd.DataFrame): Dataframe with movie_id and genre
                                    genres_df = pd.read_csv('../Data/movies.csv')

        genre: None by default. When a genre is given, it filters the output so the 
                top 5 movies recommended are from that genre. If none, no filter 
                is applied to the output.
                
    Returns:
        recommended (np.array): array with the 5 movie ids recommended
    """
    
    ## Here we can consider the option of merging the dataset before predict function
    ## And modify the predict function so the input is already the filtered dataset
    
    try:
        predict_ = model.predict(sequences = input_movie_ids)
    except:
        return []
    predicted_item_ids = np.arange(model._num_items).reshape(-1, 1)
    movies_ratings = pd.DataFrame({
        'item_ids': predicted_item_ids.reshape(len(predicted_item_ids)),
        'ratings': predict_ 
    })
    movies_ratings = movies_ratings.sort_values( by = ['ratings'], ascending = False )
    movies_ratings = movies_ratings[~movies_ratings.item_ids.isin(input_movie_ids)]

    # Merge dataframes by item_ids
    movies_ratings_genres = pd.merge(movies_ratings, genres_df, on='item_ids')

    if genre == None:
        if len(movies_ratings_genres) >= at:
            recommended = movies_ratings_genres[0:at]
        else:
            recommended = movies_ratings_genres
    else:
        recommended = movies_ratings_genres[movies_ratings_genres['genres'] == genre]
        if len(recommended) >= at:
            recommended = recommended[0:at]
        else:
            recommended = recommended
    return recommended.item_ids
