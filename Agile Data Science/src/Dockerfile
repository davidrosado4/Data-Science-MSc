FROM ubuntu:22.04

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# apt and apt-get update
RUN apt-get update && apt update && apt upgrade -y && apt-get clean
# apt and apt-get installations
RUN apt install -y vim git unzip pip

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install python 3.9
RUN apt-get update && apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.9 python3-pip 

# Install spotlight
RUN git clone https://github.com/maciejkula/spotlight.git \
    && cd spotlight \
    && python3 setup.py build \
    && python3 setup.py install

# ---------------- USER SETTING ------------------
# Add the current user to the image. Use 'id' command to find your own
ARG USER_NAME=user
ARG USER_ID=12345
ARG GROUP_NAME=falcon
ARG GROUP_ID=67890

# Update the system and install required packages
RUN apt-get update && apt-get install -y sudo

# Create a new group with the same GID as the host
RUN groupadd --gid $GROUP_ID $GROUP_NAME

# Create a new user with the same UID and GID as the host
RUN useradd --uid $USER_ID --gid $GROUP_ID --shell /bin/bash --create-home $USER_NAME

# don't require password with sudo, for convenience
# not the safest thing to do, but hopefully okay inside the container
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add the new user to the sudo group
RUN usermod -aG sudo $USER_NAME

# Set the default user to the new user
USER $USER_NAME

# -------------- END USER SETTINGS --------------

# Expose the port your application will run on 
EXPOSE 80

WORKDIR /home/${USER_NAME}

# Install other dependencies
RUN sudo apt-get update && sudo apt update && sudo apt upgrade -y && sudo apt-get clean
RUN sudo apt update && sudo apt-get install -y tmux wget

# Set the working directory in the container
ENV PROJECT_PATH=/home/${USER_NAME}/falcon_ml
WORKDIR $PROJECT_PATH