# Dockerfile for the deployment of the falcon_ml project
# docker build -t falcon-deploy .
# docker run --rm -p 8501:8501 --name falcon-deploy falcon-deploy

FROM ubuntu:22.04

# Root settings
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies from apt
RUN apt-get update && apt update && apt upgrade -y && apt-get clean
RUN apt install -y git

# Install python 3.9
RUN apt-get update && apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.9 python3-pip 

# Clone the repository
RUN git clone https://github.com/ADS-2023-TH3/falcon_ml.git

# Install requirements
RUN pip3 install --upgrade pip \
    && cd falcon_ml/src  \
    && pip3 install -r requirements.txt


# Install spotlight
RUN git clone https://github.com/maciejkula/spotlight.git \
    && cd spotlight \
    && python3 setup.py build \
    && python3 setup.py install

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables
ENV STREAMLIT_SERVER_PORT=8501

# Install other dependencies
RUN apt-get update && apt update && apt upgrade -y && apt-get clean

# Run the application
CMD ["streamlit", "run", "falcon_ml/app.py"]
