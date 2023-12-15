import sys
sys.path.append('/falcon_ml/src')
import streamlit as st
import pandas as pd
from popular_rec_model import *
from ImplicitSec_rec_model import *
import torch

from writing_functions import *
def main():
    st.title('Movie Recommender')

    # Initialize session state
    if 'success' not in st.session_state:
        st.session_state.success = False

    # Login form
    # TODO: improve it with https://blog.jcharistech.com/2020/05/30/how-to-add-a-login-section-to-streamlit-blog-app/
    data = pd.read_csv('../Data/movies.csv')
    movies = data['title'].values
    genres = ['', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy',
                'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                'Western']

    with open('../trained_models/popular_rec_model.pkl', 'rb') as file:
        popul_model = pd.read_pickle(file)

    imp_sec_model = torch.load('../trained_models/ImplicitSec_rec_model.pth')      

    #LOGIN PAGE 
    def login_page():
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:

                # Initialize session state
                st.session_state.requested_more_recommendations = 0
                st.session_state.recommendations = []
                st.session_state.more_recommendations = []
                st.session_state.ratings = {}
                st.session_state.to_remove = None

                # Connect to the users sheet
                users_worksheet = connect_to_sheet('users_sheet')
                users = users_worksheet.col_values(1)
                passwords = users_worksheet.col_values(2)

                if username in users:
                    index = users.index(username)
                    if check_hashes(password, passwords[index]):
                        # Check if the user has already rated some movies
                        st.session_state.user_ratings, st.session_state.user_ratings_indices = user_ratings(username)
                        st.session_state.username = username
                        # Display the welcome message
                        st.success("Logged in as {}".format(username))
                        st.session_state.success = True
                    else:
                        st.error("Incorrect username or password")
                        st.session_state.success = False
                else:
                    st.error("Incorrect username or password")
                    st.session_state.success = False

    #Recommender page   
    def recommender_page():
        st.header('this is the recommender')
        # Update recommendations list from previous session state ------------------------------------
        if st.session_state.to_remove != None:
            st.session_state.recommendations.remove(st.session_state.to_remove)
            st.session_state.to_remove = None

        # Preferences form ---------------------------------------------------------------------------
        with st.form("my_form"):
            selected_movies = st.multiselect("Choose Movies", movies)
            selected_genre = st.selectbox("Genre", genres)

            if selected_genre == '':
                selected_genre = None
            submitted = st.form_submit_button("Submit")

            if submitted:
                st.session_state.requested_more_recommendations = 0  # Reset requested_more_recommendations
                st.session_state.recommendations = []  # Reset recommendations
                st.session_state.more_recommendations = []  # Reset more_recommendations

                if (len(selected_movies) == 0):
                    if selected_genre == None:
                        # If no movie and no genre selected. We give the 5 films that falcons team like the most
                        st.text("No movies and no genre selected. Here are our 5 favorite movies")
                        st.session_state.recommendations = ['Pulp Fiction (1994)', 'Green Book (2018)',
                                                            'Inception (2010)', 'The Godfather (1972)',
                                                            'Kill Bill: Vol. 1 (2003)']
                    else:
                        # Popular recommender
                        movie_id_recommendations = popul_model.predict(genre=selected_genre, at=100)
                        st.session_state.recommendations = from_id_to_title(movie_id_recommendations, data)
                else:
                    # Spotlight recommender
                    # Convert title to item_id
                    titles_df = pd.DataFrame({'title': selected_movies})
                    result_df = pd.merge(titles_df, data, on='title', how='left')
                    input_movies_ids = result_df['item_ids'].values
                    # Give the prediction
                    movie_id_recommendations = predict(model=imp_sec_model, input_movie_ids=input_movies_ids,
                                                        genres_df=data, genre=selected_genre, at=100)
                    if len(movie_id_recommendations) == 0:
                        st.session_state.recommendations = [
                            "There are no available recommendations for the selected preferences"]
                    else:
                        st.session_state.recommendations = from_id_to_title(movie_id_recommendations, data)
        # --------------------------------------------------------------------------------------------
        # Display output -----------------------------------------------------------------------------
        if len(st.session_state.recommendations) > 0:  # Display recommendations obtained from the form
            st.subheader("Top 5 movies for your preferences")
            if st.session_state.recommendations[
                0] == "There are no available recommendations for the selected preferences":
                st.text("There are no available recommendations for the selected preferences")
            else:
                display_movies(st.session_state.recommendations[0:5])
        # --------------------------------------------------------------------------------------------

        # More recommendations form ------------------------------------------------------------------
        if len(st.session_state.recommendations) > 5 or st.session_state.requested_more_recommendations > 0:  # The recommendations are more than 5
            with st.form("more_recommendations_form"):
                st.subheader('More recommendations for the same preferences')

                submitted_2 = st.form_submit_button("More recommendations")

                if submitted_2:
                    if st.session_state.requested_more_recommendations > 0:  # The user has already requested more recommendations
                        # Remove the current movies in the more_recommendations list
                        del st.session_state.recommendations[5:10]

                    st.session_state.requested_more_recommendations += 1
                    # --------------------------------------------------------------------------------------------
        # Update more_recommendations independently of the form --------------------------------------
        if len(st.session_state.recommendations) > 5:  # The user has already requested more recommendations
            # Update the more_recommendations list
            if len(st.session_state.recommendations[5:]) > 5:  # If there are more than 5 recommendations left
                st.session_state.more_recommendations = st.session_state.recommendations[5:10]
            else:  # If there are less than 5 recommendations left
                st.session_state.more_recommendations = st.session_state.recommendations[5:]
        elif st.session_state.requested_more_recommendations > 0:
            st.session_state.more_recommendations = []
            st.write("There are no more recommendations. Please select new movies or a new genre")

        # --------------------------------------------------------------------------------------------
        # Display output -----------------------------------------------------------------------------
        if st.session_state.requested_more_recommendations > 0:  # Display more recommendations obtained from the form
            display_movies(st.session_state.more_recommendations)
        # --------------------------------------------------------------------------------------------

        print('session_ratings: ', st.session_state.ratings)
        print('user_ratings: ', st.session_state.user_ratings)

        # If the user has rated some movies submit the ratings to the feedback sheet using a submit button
        if len(st.session_state.ratings) > 0:
            with st.form("ratings_form"):
                submitted_3 = st.form_submit_button("Submit ratings")
                if submitted_3:
                    add_ratings(st.session_state.username, st.session_state.ratings,
                                st.session_state.user_ratings_indices)
                    # Reset ratings
                    st.session_state.ratings = {}
                    # Reset user ratings
                    st.session_state.user_ratings, st.session_state.user_ratings_indices = user_ratings(
                        st.session_state.username)
                    st.success("Ratings submitted successfully")

    #Page to 
    def history_page():
        if st.session_state.success == True:
                    # Display the last 5 movies rated by the user if there are any and is requested by the user
            if len(st.session_state.user_ratings) >= 5:
                with st.form("user_ratings_form"):
                    st.subheader("Your last 5 ratings")
                    submitted_1 = st.form_submit_button("Show ratings")
                    if submitted_1:
                        display_movies(list(st.session_state.user_ratings.keys())[-5:], st.session_state.user_ratings)

        else:
            st.write('Please log in to see your history')
    
    pages={"Log In": login_page,
            "Recommender" : recommender_page,
            "History" : history_page}
    selected_page=st.selectbox('Menu',pages.keys())
    pages[selected_page]()
    

# Additional functions
def display_movies(movies, ratings=None):
    # Display movies in a table with sliders for ratings using st.beta_columns
    for movie in movies:
        col1, col2 = st.columns(2)
        with col1:
            st.write(movie)
            if ratings is None:
                col11, col12 = st.columns(2)
                with col11:
                    replace = st.button("Replace suggestion", key=f"replace_{movie}")
                if replace:
                    if movie in st.session_state.recommendations:
                        st.session_state.to_remove = movie
                        # st.session_state.recommendations.remove(movie)
                    with col12:
                        confirm = st.button("Confirm", key=f"confirm_{movie}")
        with col2:
            if ratings is not None:
                rating = st.slider(f"Share your personal rating", 0, 5, int(ratings[movie]), key=f"rating_{movie}")
            else:
                rating = st.slider(f"Share your personal rating", 0, 5, 0, key=f"rating_{movie}")
            if rating > 0:
                st.session_state.ratings[movie] = rating


def user_ratings(username):
    feedback_worksheet = connect_to_sheet('feedback_sheet')
    feedback_users = feedback_worksheet.col_values(1)
    user_ratings_dict = {}
    if username in feedback_users:
        # index = feedback_users.index(username)
        indices = [i + 1 for i, x in enumerate(feedback_users) if x == username]
        # Get the movies and ratings that the user has already rated
        movies = [feedback_worksheet.row_values(i)[1::2][0] for i in indices]
        ratings = [feedback_worksheet.row_values(i)[2::2][0] for i in indices]
        # Create a dictionary with the movies and ratings
        user_ratings_dict = dict(zip(movies, ratings))
        # Create a dictionary that keeps track of the indices of the movies
        user_ratings_dict_indices = dict(zip(movies, indices))

    return user_ratings_dict, user_ratings_dict_indices


import hashlib


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text: 
        return True
    return False


if __name__ == '__main__':
    main()