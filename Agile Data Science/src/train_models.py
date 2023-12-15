import pickle
from popular_rec_model import *
from ImplicitSec_rec_model import *

#-----------------------------------------------------------------------------------
#-----------------------Popular recommender trainng---------------------------------
#-----------------------------------------------------------------------------------

# Train the model and store it
print('Training the popular recommender model...')
# Get data
df = get_merge_data()
# Train the model
model = TopPopRecommender()
model.fit(df)
print('Done!')
# Store it
with open('../trained_models/popular_rec_model.pkl', 'wb') as file:
    pickle.dump(model, file)

#-----------------------------------------------------------------------------------
#-----------------------Spotlight recommender trainng---------------------------------
#-----------------------------------------------------------------------------------

# Initialize and train the Implicit Sequencial model
print('Training the Implicit Sequencial model...')
df_s = load_data_to_sequences()
model_s = train_ImplicitSec_model(df_s)
print('Done!')