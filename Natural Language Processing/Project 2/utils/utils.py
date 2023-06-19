from tqdm import tqdm
from skseq.sequence_list import SequenceList
from skseq.label_dictionary import LabelDictionary
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from skseq import sequence_list_c
from keras.utils import pad_sequences
from tensorflow.keras import Model, Input
from tensorflow import keras
from tensorflow.keras.layers import InputLayer, Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, TimeDistributed
import pandas as pd

def get_data_target_sets(data):
    """
    Extracts sentences and tags from the provided data object based on sentence IDs.

    Args:
        data: A data object containing sentences and corresponding tags.

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    X = []  # Contains the sentences
    y = []  # Contains the tags

    ids = data.sentence_id.unique()  # Get unique sentence IDs from the data

    # Use tqdm to create a progress bar
    progress_bar = tqdm(ids, desc="Processing", unit="sentence")

    for sentence in progress_bar:  # Iterate over each unique sentence ID
        # Append the words for the current sentence to X
        X.append(list(data[data["sentence_id"] == sentence]["words"].values))
        # Append the tags for the current sentence to y
        y.append(list(data[data["sentence_id"] == sentence]["tags"].values))

    return X, y  # Return the lists of sentences and tags


def create_corpus(sentences, tags):
    """
    Creates a corpus by generating dictionaries for words and tags in the given sentences and tags.

    Args:
        sentences: A list of sentences.
        tags: A list of corresponding tags for the sentences.

    Returns:
        A tuple (word_dict, tag_dict, tag_dict_rev) containing dictionaries for words, tags,
        and a reversed tag dictionary.

    Example:
        sentences = [['I', 'love', 'Python'], ['Python', 'is', 'great']]
        tags = ['O', 'O', 'B']
        word_dict, tag_dict, tag_dict_rev = create_corpus(sentences, tags)
        # word_dict: {'I': 0, 'love': 1, 'Python': 2, 'is': 3, 'great': 4}
        # tag_dict: {'O': 0, 'B': 1}
        # tag_dict_rev: {0: 'O', 1: 'B'}
    """
    word_dict = {}  # Contains unique words with corresponding indices
    tag_dict = {}  # Contains unique tags with corresponding indices

    # Generate word dictionary
    for sentence in sentences:
        for word in sentence:
            if word not in word_dict:
                word_dict[word] = len(word_dict)

    # Generate tag dictionary
    for tag_list in tags:
        for tag in tag_list:
            if tag not in tag_dict:
                tag_dict[tag] = len(tag_dict)

    tag_dict_rev = {v: k for k, v in tag_dict.items()}  # Reverse tag dictionary

    return word_dict, tag_dict, tag_dict_rev


def create_sequence_listC(word_dict, tag_dict, X, y):
    """
    Creates a sequence list object by adding sequences from X and y to it using cython.

    Args:
        word_dict: A dictionary mapping words to their corresponding indices.
        tag_dict: A dictionary mapping tags to their corresponding indices.
        X: A list of input sequences (sentences).
        y: A list of corresponding target sequences (tags).

    Returns:
        A sequence list object populated with sequences from X and y.
    """
    seq = sequence_list_c.SequenceListC(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # Use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Adding sequences", unit="sequence")

    for i in progress_bar:
        # Add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq
def create_sequence_list(word_dict, tag_dict, X, y):
    """
    Creates a sequence list object by adding sequences from X and y to it without cython.

    Args:
        word_dict: A dictionary mapping words to their corresponding indices.
        tag_dict: A dictionary mapping tags to their corresponding indices.
        X: A list of input sequences (sentences).
        y: A list of corresponding target sequences (tags).

    Returns:
        A sequence list object populated with sequences from X and y.
    """
    seq = SequenceList(LabelDictionary(word_dict), LabelDictionary(tag_dict))

    # Use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Adding sequences", unit="sequence")

    for i in progress_bar:
        # Add the sequence (X[i], y[i]) to the sequence list
        seq.add_sequence(X[i], y[i], LabelDictionary(word_dict), LabelDictionary(tag_dict))

    return seq

def show_features(feature_mapper, seq, feature_type=["Initial features", "Transition features", "Final features", "Emission features"]):
    """
    Displays the features extracted from a sequence using a feature mapper.

    Args:
        feature_mapper: An object responsible for mapping feature IDs to feature names.
        seq: A sequence object containing the input sequence.
        feature_type: Optional. A list of feature types to display. Default is ["Initial features", "Transition features", "Final features", "Emission features"].

    Returns:
        None
    """
    inv_feature_dict = {word: pos for pos, word in feature_mapper.feature_dict.items()}

    for feat, feat_ids in enumerate(feature_mapper.get_sequence_features(seq)):
        print(feature_type[feat])  # Print the current feature type

        for id_list in feat_ids:
            for k, id_val in enumerate(id_list):
                print(id_list, inv_feature_dict[id_val])  # Print the feature IDs and their corresponding names

        print("\n")  # Add a newline after printing all features of a certain type


def get_tiny_test():
    """
    Creates a tiny test dataset.

    Args:
        None

    Returns:
        A tuple (X, y) containing lists of sentences (X) and tags (y).
    """
    X = [['The programmers from Barcelona might write a sentence without a spell checker . '],
         ['The programmers from Barchelona cannot write a sentence without a spell checker . '],
         ['Jack London went to Parris . '],
         ['Jack London went to Paris . '],
         ['Bill gates and Steve jobs never though Microsoft would become such a big company . '],
         ['Bill Gates and Steve Jobs never though Microsof would become such a big company . '],
         ['The president of U.S.A though they could win the war . '],
         ['The president of the United States of America though they could win the war . '],
         ['The king of Saudi Arabia wanted total control . '],
         ['Robin does not want to go to Saudi Arabia . '],
         ['Apple is a great company . '],
         ['I really love apples and oranges . '],
         ['Alice and Henry went to the Microsoft store to buy a new computer during their trip to New York . ']]

    y = [['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'O', 'B-geo', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'I-per', 'O', 'B-per', 'I-per', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'I-geo', 'I-geo', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'B-geo', 'I-geo', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'I-geo', 'O'],
            ['B-org', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
            ['B-per', 'O', 'B-per', 'O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-geo',
             'I-geo', 'O']]

    return [i[0].split() for i in X], y


def predict_SP(model, X):
    """
    Predicts the tags for the input sequences using a StructuredPerceptron model.

    Args:
        model: A trained StructuredPerceptron model.
        X: A list of input sequences (sentences).

    Returns:
        A list of predicted tags for the input sequences.
    """
    y_pred = []

    # Use tqdm to create a progress bar
    progress_bar = tqdm(range(len(X)), desc="Predicting tags", unit="sequence")

    for i in progress_bar:
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    y_pred = [np.ndarray.tolist(array) for array in y_pred]
    y_pred = np.concatenate(y_pred).ravel().tolist()

    return y_pred


def accuracy(true, pred):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    # Get indexes of those that are not 'O'
    idx = [i for i, x in enumerate(true) if x != 'O']

    # Get the true and predicted tags for those indexes
    true = [true[i] for i in idx]
    pred = [pred[i] for i in idx]

    # Use sklearn's accuracy_score to compute the accuracy
    return accuracy_score(true, pred)


def plot_confusion_matrix(true, pred, tag_dict_rev):
    """
    Plots a confusion matrix using a heatmap.

    Args:
        true: A list or array of true labels.
        pred: A list or array of predicted labels.
        tag_dict_rev: A dictionary mapping tag values to their corresponding labels.

    Returns:
        None
    """
    # Get all unique tag values from true and pred lists
    unique_tags = np.unique(np.concatenate((true, pred)))

    # Create a tick label list with all unique tags
    tick_labels = [tag_dict_rev.get(tag, tag) for tag in unique_tags]

    # Get the confusion matrix
    cm = confusion_matrix(true, pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def f1_score_weighted(true, pred):
    """
    Computes the weighted F1 score based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The weighted F1 score.
    """
    # Get the weighted F1 score using sklearn's f1_score function
    return f1_score(true, pred, average='weighted')


def evaluate(true, pred, tag_dict_rev):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    # Compute the accuracy and F1 score using predefined functions
    acc = accuracy(true, pred)
    f1 = f1_score_weighted(true, pred)

    # Print the evaluation results
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    # Plot the confusion matrix
    plot_confusion_matrix(true, pred, tag_dict_rev)


def print_tiny_test_prediction(X, model, tag_dict_rev):
    """
    Prints the predicted tags for each input sequence.

    Args:
        X: A list of input sequences.
        model: The trained model used for prediction.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    y_pred = []
    for i in range(len(X)):
        # Predict the tags for the current input sequence
        predicted_tag = model.predict_tags_given_words(X[i])
        y_pred.append(predicted_tag)

    for i in range(len(X)):
        sentence = X[i]
        tag_list = y_pred[i]
        prediction = ''
        for j in range(len(sentence)):
            prediction += sentence[j] + "/" + tag_dict_rev[tag_list[j]] + " "

        print(prediction + "\n")

#----------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- DL Approach ------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def preprocess_BiLSTM_train_data(df, max_len=128):
    """
    Preprocesses training data for a BiLSTM model.

    Args:
        df: DataFrame containing the training data.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        X: Preprocessed input data (padded sequences of word indices).
        y: Preprocessed target data (padded sequences of tag indices).
        num_words: Total number of unique words in the training data.
        num_tags: Total number of unique tags in the training data.
        word2idx: Dictionary mapping words to their corresponding indices.
        tag2idx: Dictionary mapping tags to their corresponding indices.
    """

    # Fill missing values in the "sentence_id" column with the previous non-null value
    df.loc[:, "sentence_id"] = df["sentence_id"].fillna(method="ffill")

    # Define a lambda function to aggregate words and tags into tuples
    agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                 s["tags"].values.tolist())]

    # Group the dataframe by "sentence_id" and apply the aggregation function
    sentences = df.groupby('sentence_id').apply(agg_func).tolist()

    # Create a list of unique words in the training data and add an "ENDPAD" token
    words = list(dict.fromkeys(df["words"].values))
    words.append("ENDPAD")
    num_words = len(words)

    # Create a list of unique tags in the training data
    tags = list(dict.fromkeys(df["tags"].values))
    num_tags = len(tags)

    # Create a word-to-index dictionary
    word2idx = {w: i + 1 for i, w in enumerate(words)}  # Index starts from 1

    # Create a tag-to-index dictionary
    tag2idx = {t: i for i, t in enumerate(tags)}

    # Convert words to their corresponding indices using word2idx dictionary
    X = [[word2idx[w[0]] for w in s] for s in sentences]

    # Pad the sequences of word indices to a fixed length
    # Use the value num_words-1 for padding
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

    # Convert tags to their corresponding indices using tag2idx dictionary
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # Pad the sequences of tag indices to a fixed length
    # Use the value of tag2idx["O"] for padding
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    # Return the processed data: X, y, num_words, num_tags, word2idx, tag2idx
    return X, y, num_words, num_tags, word2idx, tag2idx


def preprocess_BiLSTM_test_data(df, word2idx, tag2idx, num_words, max_len=128):
    """
    Preprocesses test data for a BiLSTM model.

    Args:
        df: DataFrame containing the test data.
        word2idx: Dictionary mapping words to their corresponding indices.
        tag2idx: Dictionary mapping tags to their corresponding indices.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        X: Preprocessed input data (padded sequences of word indices).
        y: Preprocessed target data (padded sequences of tag indices).
    """

    # Fill missing values in the "sentence_id" column with the previous non-null value
    df.loc[:, "sentence_id"] = df["sentence_id"].fillna(method="ffill")

    # Define a lambda function to aggregate words and tags into tuples
    agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                 s["tags"].values.tolist())]

    # Group the dataframe by "sentence_id" and apply the aggregation function
    sentences = df.groupby('sentence_id').apply(agg_func).tolist()

    # Convert words to their corresponding indices using word2idx dictionary
    # Ignore words not found in the dictionary (use 0 as the index)
    X = [[word2idx.get(w[0], 0) for w in s if w[0] in word2idx.keys()] for s in sentences]

    # Pad the sequences of word indices to a fixed length
    # Use the value num_words-1 for padding
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)

    # Convert tags to their corresponding indices using tag2idx dictionary
    y = [[tag2idx[w[1]] for w in s] for s in sentences]

    # Pad the sequences of tag indices to a fixed length
    # Use the value of tag2idx["O"] for padding
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    # Return the processed data: X, y
    return X, y


def create_BiLSTM_model(num_words, max_len=128):
    """
    Creates a BiLSTM model.

    Args:
        num_words: Total number of unique words in the input data.
        max_len: Maximum length of sequences (default: 128).

    Returns:
        model: BiLSTM model.
    """

    # Create a sequential model
    model = keras.Sequential()

    # Add an input layer with the specified input shape
    model.add(InputLayer((max_len)))

    # Add an embedding layer with the specified input dimension, output dimension, and input length
    model.add(Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len))

    # Add a spatial dropout layer with a dropout rate of 0.1
    model.add(SpatialDropout1D(0.1))

    # Add a bidirectional LSTM layer with 100 units, returning sequences, and a recurrent dropout of 0.1
    model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model
def accuracy_lstm(true, pred):
    """
    Computes the accuracy of predicted tags compared to true tags, excluding instances where true[i] == 'O'.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.

    Returns:
        The accuracy score, which measures the proportion of correct predictions.
    """
    # Get indexes of those that are not 'O'
    idx = [i for i, x in enumerate(true) if x != 0]

    # Get the true and predicted tags for those indexes
    true = [true[i] for i in idx]
    pred = [pred[i] for i in idx]

    # Use sklearn's accuracy_score to compute the accuracy
    return accuracy_score(true, pred)


def evaluate_lstm(true, pred, tag_dict_rev):
    """
    Computes and prints evaluation metrics and displays a confusion matrix based on the true and predicted tags.

    Args:
        true: A list of true tags.
        pred: A list of predicted tags.
        tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    Returns:
        None
    """
    # Compute the accuracy and F1 score using predefined functions
    acc = accuracy_lstm(true, pred)
    f1 = f1_score_weighted(true, pred)

    # Print the evaluation results
    print('Accuracy: {:.4f}'.format(acc))
    print('F1 Score: {:.4f}'.format(f1))

    # Plot the confusion matrix
    plot_confusion_matrix(true, pred, tag_dict_rev)

def predict_lstm(model, X):
    """
    Predicts labels using an LSTM model.

    Arguments:
    - model: The LSTM model used for prediction.
    - X: The input data to be used for prediction.

    Returns:
    - y_train_pred: The predicted labels based on the input data.
    """

    # Predict using the LSTM model
    y_train_pred = model.predict(X)

    # Get the index with the maximum probability as the predicted label
    y_train_pred = np.argmax(y_train_pred, axis=-1)

    # Limit the predicted labels to a maximum of 16
    y_train_pred[y_train_pred > 16] = 16

    # Return the predicted labels
    return y_train_pred

def get_tiny_test_lstm():
    """
    Creates a dataframe with the data from X_tiny and y_tiny.

    Returns:
    - df: A dataframe containing the sentence IDs, words, and tags.
    """

    # Initialize empty lists for each column
    sentence_ids = []
    words = []
    tags = []

    # Get X_tiny and y_tiny
    X_tiny, y_tiny = get_tiny_test()

    # Iterate over each sentence and its corresponding tags
    for i, (sentence_tokens, tag_tokens) in enumerate(zip(X_tiny, y_tiny)):
        # Extract words and tags for the current sentence
        sentence = sentence_tokens
        tags_list = tag_tokens

        # Append words and tags to the respective lists
        words.extend(sentence)
        tags.extend(tags_list)
        sentence_ids.extend([i + 1] * len(sentence))  # Assign sentence ID to each word

    # Create a dictionary with the data
    data = {
        'sentence_id': sentence_ids,
        'words': words,
        'tags': tags
    }

    # Create the dataframe
    df = pd.DataFrame(data)
    return df
def print_tiny_test_prediction_lstm(X_tiny, y_tiny_pred, word2idx, tag_dict_rev):
    """
    Prints the predictions for the X_tiny dataset using the LSTM model.

    Arguments:
    - X_tiny: The input data for prediction.
    - y_tiny_pred: The predicted tags for the input data.
    - word2idx: A dictionary mapping words to their corresponding indices.
    - tag_dict_rev: A dictionary mapping tag indexes to tag labels.

    """

    # Create a reversed vocabulary dictionary for mapping indices back to words
    reversed_vocabulary = {value: key for key, value in word2idx.items()}

    # Iterate over the sentences and their predicted tags
    for i in range(0, 12):
        sentence = X_tiny[i]
        tags = y_tiny_pred[i]
        sentence_short = []

        # Create a shortened sentence by stopping at the first occurrence of a point (value 22)
        for i in range(0, len(sentence)):
            sentence_short.append(sentence[i])
            if sentence_short[i] == 22:
                break

        # Convert the sentence indices to word vectors using the reversed vocabulary
        word_vector = [reversed_vocabulary[position] for position in sentence_short]
        tags = tags[0:len(word_vector)]

        # Convert the tag indices to tag vectors using the tag dictionary
        tags_vector = [tag_dict_rev[position] for position in tags]

        # Print the token and its corresponding tag
        for token, tag in zip(word_vector, tags_vector):
            print(f'{token}/{tag}', end=' ')
        print('\n')



