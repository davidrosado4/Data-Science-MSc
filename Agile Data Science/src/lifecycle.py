from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.synthetic import generate_sequential
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.datasets.movielens import get_movielens_dataset
import numpy as np
import mlflow


def load_data_to_sequences(variant='20M',
                           max_sequence_length=200,
                           min_sequence_length=20,
                           step_size=200, df_split=False):
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

# Mlflow experiments
# Set experiment name

experiment_name = "Mlflow Experiment"
mlflow.set_experiment(experiment_name)

# Load data
train, test = load_data_to_sequences(df_split=True)

# Set parameters
n_iter = 7
representation = 'lstm'
loss = 'hinge'

# Create model
model = ImplicitSequenceModel(n_iter=n_iter,
                              representation=representation,
                              loss=loss)
print("Fitting the model...")
# Train the model
model.fit(train, verbose=True)
print("Computing MRR...")
# Evaluate the model
mrr_list = sequence_mrr_score(model, test)
mrr_mean = np.mean(mrr_list)

# Log parameters
with mlflow.start_run():
    mlflow.log_param("n_iter", n_iter)
    mlflow.log_param("representation", representation)
    mlflow.log_param("loss", loss)


    # Log the mean of MRR scores
    mlflow.log_metric("mrr_mean", mrr_mean)
