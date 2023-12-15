The assignment was based in a final project. We created a web app recommendation.

**Final mark:**

---
# Movie Recommender Web App

<div style="text-align:center; margin-bottom:20px;">
  <img alt="Captura de pantalla 2023-12-14 a las 9 54 42" src="https://github.com/ADS-2023-TH3/falcon_ml/assets/114001733/c19c3b3d-91ea-4b79-88bd-0969811447be" width="900">
</div>

<br><br>
Welcome to our Movie Recommender Web App! This web application allows users to input a movie title and receive personalized movie recommendations based on our advanced recommendation algorithm. It's powered by cutting-edge data science techniques, making your movie-watching experience even better!
<br><br>
<div style="text-align:center; margin-bottom:20px;">
  <img alt="Captura de pantalla 2023-12-14 a las 9 57 29" src="https://github.com/ADS-2023-TH3/falcon_ml/assets/114001733/a157884e-2538-41e7-ac46-c148ce68aab2" width="900">
</div>
<br>

## Features

- **User-Friendly Interface:** Simple and intuitive interface for easy navigation.
- **Personalized Recommendations:** Get tailored movie suggestions based on your input.
- **Data-Driven Recommendations:** Powered by our powerful data science model, ensuring high-quality suggestions.
- **Quick and Responsive:** Fast processing to provide instant recommendations.

## How to Use

1. **Input Movie Title:** Enter the title of a movie you like into the input field.
2. **Receive Recommendations:** Instantly get a list of top movie recommendations based on your input.

## Tech Stack

- **Backend:** Python, Streamlit 
- **Frontend:** HTML, CSS, JavaScript 
- **Data Science:** Pytorch, Scikit-learn, Pandas 

## Setup Instructions

1. Download the [Dockerfile](https://github.com/ADS-2023-TH3/falcon_ml/blob/main/Dockerfile). Or clone the repository:

```
git clone https://github.com/ADS-2023-TH3/falcon_ml.git && cd falcon_ml
```

2. Build the Docker image:

```
docker build --no-cache -t falcon-deploy .
```

3. Run the Docker image at port 8501:

```
docker run --rm -p 8501:8501 --name falcon-deploy falcon-deploy
```

4. Open the web app in your browser: [http://localhost:8501](http://localhost:8501)

## Contributing

We welcome contributions from the community! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.
