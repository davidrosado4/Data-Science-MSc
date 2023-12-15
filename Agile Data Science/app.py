import sys
sys.path.append('/falcon_ml/src')
import streamlit as st
import pandas as pd
from popular_rec_model import *
from ImplicitSec_rec_model import *
import torch
import urllib.parse

from writing_functions import *
import base64

def main():
    set_background('/falcon_ml/assets/banner-1.png')


    # Display the image
    st.image('/falcon_ml/assets/falcon.png', width=700)
    # Initialize session state
    if 'success' not in st.session_state:
        st.session_state.success = False

    # Login form
    # TODO: improve it with https://blog.jcharistech.com/2020/05/30/how-to-add-a-login-section-to-streamlit-blog-app/
    
    # Login page
    def login_page():
        with st.form("login_form"):
            st.subheader("Log In")
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
        
        if st.session_state.success == True:
            st.button("Continue to the web app")

    # Recommender page
    def recommender_page():            
        if st.session_state.success:
            data = pd.read_csv('/falcon_ml/Data/movies.csv')
            movies = data['title'].values
            genres = ['', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                    'Western']

            with open('/falcon_ml/trained_models/popular_rec_model.pkl', 'rb') as file:
                popul_model = pd.read_pickle(file)

            imp_sec_model = torch.load('/falcon_ml/trained_models/ImplicitSec_rec_model.pth')      

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
                    st.session_state.requested_more_recommendations = 0   # Reset requested_more_recommendations
                    st.session_state.recommendations = []   # Reset recommendations
                    st.session_state.more_recommendations = []   # Reset more_recommendations

                    if (len(selected_movies) == 0):
                        if selected_genre == None:
                            # If no movie and no genre selected. We give the 5 films that falcons team like the most
                            st.text("No movies and no genre selected. Here are our 5 favorite movies")
                            st.session_state.recommendations = ['Pulp Fiction (1994)','Green Book (2018)',
                                                                'Inception (2010)', 'The Godfather (1972)', 'Kill Bill: Vol. 1 (2003)']
                        else:
                            # Popular recommender
                            movie_id_recommendations = popul_model.predict(genre=selected_genre, at=100)
                            st.session_state.recommendations = movies_with_ratings_from_ids(movie_id_recommendations, data)
                            #st.session_state.recommendations = from_id_to_title(movie_id_recommendations, data)
                    else:
                        # Spotlight recommender
                        # Convert title to item_id
                        titles_df = pd.DataFrame({'title': selected_movies})
                        result_df = pd.merge(titles_df, data, on='title', how='left')
                        input_movies_ids = result_df['item_ids'].values
                        # Give the prediction
                        movie_id_recommendations = predict(model=imp_sec_model, input_movie_ids=input_movies_ids,
                                                        genres_df = data,genre=selected_genre,at=100)
                        if len(movie_id_recommendations) == 0:
                            st.session_state.recommendations = ["There are no available recommendations for the selected preferences"]
                        else:
                            st.session_state.recommendations = movies_with_ratings_from_ids(movie_id_recommendations, data)
                            #st.session_state.recommendations = from_id_to_title(movie_id_recommendations, data)
            # --------------------------------------------------------------------------------------------
            # Display output -----------------------------------------------------------------------------
            if len(st.session_state.recommendations) > 0:   # Display recommendations obtained from the form
                st.subheader("Top 5 movies for your preferences")
                if st.session_state.recommendations[0] == "There are no available recommendations for the selected preferences":
                    st.text("There are no available recommendations for the selected preferences")
                else:
                    display_movies(st.session_state.recommendations[0:5])
            # --------------------------------------------------------------------------------------------

            # More recommendations form ------------------------------------------------------------------
            if len(st.session_state.recommendations) > 5 or st.session_state.requested_more_recommendations > 0:   # The recommendations are more than 5        
                with st.form("more_recommendations_form"):
                    st.subheader('More recommendations for the same preferences')
                    
                    submitted_2 = st.form_submit_button("More recommendations")
                        
                    if submitted_2:
                        if st.session_state.requested_more_recommendations > 0:   # The user has already requested more recommendations
                            # Remove the current movies in the more_recommendations list
                            del st.session_state.recommendations[5:10]
                            
                        st.session_state.requested_more_recommendations += 1           
            # --------------------------------------------------------------------------------------------
            # Update more_recommendations independently of the form --------------------------------------
            if len(st.session_state.recommendations) > 5:   # The user has already requested more recommendations
                # Update the more_recommendations list
                if len(st.session_state.recommendations[5:]) > 5:   # If there are more than 5 recommendations left
                    st.session_state.more_recommendations = st.session_state.recommendations[5:10]
                else:   # If there are less than 5 recommendations left
                    st.session_state.more_recommendations = st.session_state.recommendations[5:]
            elif st.session_state.requested_more_recommendations > 0:
                st.session_state.more_recommendations = []
                st.write("There are no more recommendations. Please select new movies or a new genre")

            # --------------------------------------------------------------------------------------------
            # Display output -----------------------------------------------------------------------------
            if st.session_state.requested_more_recommendations > 0:   # Display more recommendations obtained from the form
                display_movies(st.session_state.more_recommendations)
            # --------------------------------------------------------------------------------------------

            print('session_ratings: ', st.session_state.ratings)
            print('user_ratings: ', st.session_state.user_ratings)

            # If the user has rated some movies submit the ratings to the feedback sheet using a submit button
            if len(st.session_state.ratings) > 0:
                with st.form("ratings_form"):
                    submitted_3 = st.form_submit_button("Submit ratings")
                    if submitted_3:
                        add_ratings(st.session_state.username, st.session_state.ratings, st.session_state.user_ratings_indices)
                        # Reset ratings
                        st.session_state.ratings = {}
                        # Reset user ratings
                        st.session_state.user_ratings, st.session_state.user_ratings_indices = user_ratings(st.session_state.username)
                        st.success("Ratings submitted successfully")
        
        else:
            st.write('Please log in to see your history')

    def history_page():
        if st.session_state.success == True:
            # Display the last 5 movies rated by the user if there are any and is requested by the user
            if len(st.session_state.user_ratings) >= 5:
                with st.form("user_ratings_form"):
                    st.subheader("Your last 5 ratings")
                    submitted_1 = st.form_submit_button("Show ratings") 
                    if submitted_1:
                        # TODO: bug fix, pass the list of dictionaries for the movies, instead of just the list of titles
                        movies = list(st.session_state.user_ratings.keys())[-5:]
                        movies_dicts = []
                        for movie in movies:
                            movies_dicts.append({'title': movie, 'rating': st.session_state.user_ratings[movie]})
                        display_movies(movies_dicts, st.session_state.user_ratings)
            else:
                st.write("You have not rated any movies yet")
        else:
            st.write('Please log in to see your history')

    def about_page():
        st.title("About Falcon ML")

        st.markdown(
            """
            ## Introduction

            Welcome to Falcon ML, an exciting movie recommender web app developed by a dedicated team of seven individuals as part of an Agile Data Science course at University of Barcelona. Falcon ML is designed to provide personalized movie recommendations based on user preferences, utilizing state-of-the-art machine learning algorithms to enhance the movie-watching experience.

            ## Project Overview

            ### Objective

            The primary goal of Falcon ML is to offer users a seamless and enjoyable way to discover new movies tailored to their tastes. By leveraging cutting-edge data science techniques, Falcon ML aims to deliver accurate and personalized movie recommendations, transforming the way users explore and enjoy cinematic content.

            ### Key Features

            - **Personalized Recommendations**: Falcon ML analyzes user preferences and behavior to generate movie suggestions that align with individual tastes.
            
            - **Dynamic Genre Selection**: Users can explore movies from various genres, ensuring a diverse range of recommendations to suit different moods and interests.

            - **Implicit and Explicit Feedback**: Falcon ML considers both explicit user ratings and implicit preferences to fine-tune its recommendation engine, providing a comprehensive and nuanced experience.

            - **Agile Development**: The project is developed following Agile principles, allowing the team to adapt and iterate efficiently, ensuring the app evolves based on user feedback and changing requirements.

            ## Meet the Team

            Our talented team of seven members has collaborated to bring Falcon ML to life. Each team member contributes their unique skills and expertise, fostering a collaborative environment for innovation and success.

            - [Leonardo Bocchi](https://github.com/leobcc)
            - [Carmen Casas](https://github.com/ccasash)
            - [Flàvia Ferrús](https://github.com/flaviaferrus)
            - [Arturo Fredes](https://github.com/arturofredes)
            - [Àlex Pujol](https://github.com/alex-pv01)
            - [David Rosado](https://github.com/davidrosado4)
            - [Jaume Sanchez](https://github.com/jshz12)

            ## Technologies Used

            Falcon ML is built using a robust stack of technologies, including:

            - **Python**: The core programming language for data analysis and backend development.
            - **Streamlit**: Powering the interactive and user-friendly web interface.
            - **Pandas**: Handling and processing movie data efficiently.
            - **Machine Learning Models**: Incorporating recommendation models for accurate suggestions.

            ## Feedback and Contributions

            We value your feedback! Falcon ML is an ongoing project, and we welcome contributions, suggestions, and bug reports from the community. Feel free to reach out to us through [GitHub Issues](https://github.com/ADS-2023-TH3/falcon_ml/issues) to share your thoughts.

            Thank you for exploring Falcon ML – your go-to movie recommender for a personalized cinematic journey!
            """
        )

    def team_page():
        st.title("About the Falcon ML Team")

        st.write(
            "Welcome to the Falcon ML project! We are a team of passionate individuals working on a movie recommender web app called Falcon ML."
        )

        st.header("Meet the Team")

        st.subheader("1. Scrum Master: Carmen Casas")
        st.write(
            "Carmen is an enthusiastic data scientist with a keen interest in machine learning and recommender systems. "
            "As the Scrum Master, she provides guidance and ensures smooth collaboration among team members."
        )

        st.subheader("2. Data Scientists: Flàvia Ferrús & David Rosado")
        st.write(
            "Flàvia and David possess expertise in data science, with a focus on data preprocessing and feature engineering. "
            "They play a crucial role in preparing and analyzing the dataset for the recommender system, as well as selecting the optimal model for training it."
        )

        st.subheader("3. Frontend Developer: Leonardo Bocchi")
        st.write(
            "Leonardo is an experienced frontend developer responsible for creating an engaging and user-friendly web interface. "
            "He focuses on building a seamless user experience for Falcon ML."
        )

        st.subheader("4. Backend Developers: Arturo Fredes & Jaume Sanchez")
        st.write(
            "Arturo and Jaume are experienced backend developers who specialize in constructing scalable and efficient server-side components."
            "They ensure the backend of Falcon ML operates smoothly and handles user requests effectively."
        )

        st.subheader("5. DevOps Engineer: Àlex Pujol")
        st.write(
            "Àlex is a skilled DevOps engineer who manages the deployment and infrastructure of Falcon ML. "
            "He ensures the application runs smoothly and efficiently in a production environment."
        )

        st.header("Project Overview")

        st.write(
            "Falcon ML is a movie recommender web app developed by our team. "
            "Our goal is to provide users with personalized movie recommendations based on their preferences and viewing history."
        )
    def contact_page():
        st.header("Contact Us")

        st.write(
            "If you have any questions, feedback, or inquiries, feel free to get in touch with us! We value your input and are here to assist you."
        )

        st.subheader("Contact Form")

        # Create a form to gather user input
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            submitted = st.form_submit_button("Submit")

            if submitted:
                # Customize the behavior upon form submission (e.g., send an email, save data)
                # For now, print the submitted information
                st.success(f"Thank you, {name}! Your message has been submitted.")

        st.header("GitHub Repository")

        st.write(
            "Our project is hosted on GitHub. You can contribute, report issues, or explore the codebase on our [GitHub repository](https://github.com/ADS-2023-TH3/falcon_ml)."
        )
        st.write(
            "If you encounter any issues or have feature requests, please open an issue on our GitHub repository. Your feedback is important to us, and we appreciate your contributions."
        )
    
    # Display the pages -----------------------------------------------------------------------
    if 'selected_lp_menu_option' not in st.session_state:
        st.session_state.selected_lp_menu_option = "Web App"
    lp_menu_options = {"Web App": None,
                        "About the Project" : about_page,
                        "The Team" : team_page,
                        "Contact Us" : contact_page}
    st.session_state.selected_lp_menu_option = st.sidebar.selectbox("Learn More", lp_menu_options.keys(), key="learn_more")
    if st.session_state.selected_lp_menu_option != "Web App":
        lp_menu_options[st.session_state.selected_lp_menu_option]()

    if st.session_state.success == False and st.session_state.selected_lp_menu_option == "Web App":   # If the user is not logged in, show the login page
        wa_menu_options = {"Log In" : login_page,
                        "Recommender" : recommender_page,
                        "User History" : history_page}
        selected_wa_menu_option = st.sidebar.selectbox("Web App Menu", wa_menu_options.keys())
        if selected_wa_menu_option == "Log In":
            login_page()
        else:
            st.write('Please log in to use the web app')   
            login_page()
    elif st.session_state.success == True and st.session_state.selected_lp_menu_option == "Web App":   
        wa_menu_options = {"Recommender" : recommender_page,
                        "User History" : history_page}
        selected_wa_menu_option = st.sidebar.selectbox("Web App Menu", wa_menu_options.keys())
        wa_menu_options[selected_wa_menu_option]()
                
        

# Functions -----------------------------------------------------------------------------------------
def movies_with_ratings_from_ids(movie_ids, data):
    recommendations_with_ratings = []
    movie_titles = from_id_to_title(movie_ids, data)
    ratings = ordered_ratings_films(movie_ids)
    
    for movie, rating in zip(movie_titles, ratings):
        movie_info = {'title': movie, 'rating': rating}
        recommendations_with_ratings.append(movie_info)
    
    return recommendations_with_ratings
    
def display_movies(movies, ratings=None):
    # Display movies in a table with sliders for ratings using st.beta_columns
    for movie in movies:
        col1, col2 = st.columns(2)
        with col1:
            imdb_url = imdb_search_url(movie['title'])
            st.write(movie['title'], f"[[IMDb]]({imdb_url})")
            if ratings is None:
                col11, col12 = st.columns(2)
                with col11:
                    replace = st.button("Replace suggestion", key=f"replace_{movie['title']}")
                if replace:
                    if movie in st.session_state.recommendations:
                        st.session_state.to_remove = movie
                        #st.session_state.recommendations.remove(movie)
                    with col12:
                        confirm = st.button("Confirm", key=f"confirm_{movie['title']}")
        with col2:
            if ratings is not None:
                 rating = st.slider(f"Share your personal rating, [Our rating: {get_star_html(movie['rating'])}]", 0, 5, int(ratings[movie['title']]), key=f"rating_{movie['title']}")
            else:
                rating = st.slider(f"Share your personal rating, [Our rating: {get_star_html(movie['rating'])}]", 0, 5, 0, key=f"rating_{movie['title']}")
            if rating > 0:
                st.session_state.ratings[movie['title']] = rating

def get_star_html(rating):
    # Create HTML string for star rating display with red color
    stars = int(rating)
    red_star_html = ':star:'
    star_html = red_star_html * stars
    return star_html

def user_ratings(username):
    feedback_worksheet = connect_to_sheet('feedback_sheet')
    feedback_users = feedback_worksheet.col_values(1)
    user_ratings_dict = {}
    if username in feedback_users:
        #index = feedback_users.index(username)
        indices = [i+1 for i, x in enumerate(feedback_users) if x == username]
        # Get the movies and ratings that the user has already rated
        movies = [feedback_worksheet.row_values(i)[1::2][0] for i in indices]
        ratings = [feedback_worksheet.row_values(i)[2::2][0] for i in indices]
        # Create a dictionary with the movies and ratings
        user_ratings_dict = dict(zip(movies, ratings))
        # Create a dictionary that keeps track of the indices of the movies
        user_ratings_dict_indices = dict(zip(movies, indices))

    return user_ratings_dict, user_ratings_dict_indices

def imdb_search_url(movie_title):
    base_url = "https://www.imdb.com/find?q="
    encoded_title = urllib.parse.quote_plus(movie_title)
    return f"{base_url}{encoded_title}"

import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return True
	return False

# Background image -----------------------------------------------------------------------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

if __name__ == '__main__':
    main()