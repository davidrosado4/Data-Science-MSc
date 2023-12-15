# Instuctions to run the container in development mode
We can do this so everyone works on the same environment.

First clone the repo to your local machine:
```
git clone git@github.com:ADS-2023-TH3/falcon_ml.git
```

Then cd into the src directory:
```
cd falcon_ml/src
```

To build the docker image, run the following command:
```
docker build -t falcon-dev .
```

Go to the main directory:
```
cd ..
```

To run the docker image, run the following command: </br>
Ubuntu/Linux:
```
docker run -it -p 5000:5000 -v $(pwd):/home/user/falcon_ml falcon-dev
```
For Windows:
```
docker run -it -p 5000:5000 -v ${PWD}:/home/user/falcon_ml falcon-dev
```

The ```-v``` flag is used to mount the current directory to the docker container. This means that any changes you make to the code will be reflected in the container.

If you want to use git from the container make sure your credentials are properly set up. You can do this by running the following commands:
```
git config --global user.name "Your Name"
```
```
git config --global user.email "your@mail.com
```

Is important to keep track of the updated requirements and rebuild the image if needed to install new dependencies. To do this, run the following command at the ```src``` directory:
```
docker build -t falcon-dev .
```
