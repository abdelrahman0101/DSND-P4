[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Dogs represent one of the most common domestic animals since ancient history. They have many breeds known that spread all over the world. The AKC lists over 150 known dog breeds. This project presents some experiments conducted to build an automated dog breed classifier that takes an image, detects whether it includes a dog or not, and classifies the detected dog breed. The implemented algorithm can also detect human faces and suggest the most similar dog breed to a given human photo.
The project uses a deep neural network model implemented using Keras with a TensorFlow back-end. This model is trained on a large labelled data-set of dog images.


## Motivation
Classifying dog breeds from images is a typical computer vision problem that might be useful in many applications. It's sometimes difficult for even humans to differentiate between some similar dog breeds. Thus, having a software that can automatically detect dogs and classify dog breeds may help automate this task.

This project was implemented as part of Udacity's Data Scientist Nanodegree program.


## Libraries
This project uses the following python libraries for implementation:
* __Sklearn:__ for basic machine learning tasks.
* __Keras:__ To build, train, and evaluate neural network models based on the Tensorflow backend.
* __Numpy:__ for representing data  in multidimensional arrays during processing.
* __Pandas:__ For representing data in dataframes. Used during initial exploratory analysis.
* __Matplotlib:__ For creating visualizations on data statistics, model performance, and classification output.



## Project Structure
The main files in this project are:
* __bottleneck_features:__ this directory includes the VGG16 bottleneck features for the VGG16 model.
* __haarcascades:__ this directory contains the pretrained OpenCV models for detecting human faces.
* __images:__ some images used in the jupyter notebook as part of the project instructions.
* __requirements:__ software requirements for running the project on different platforms.
* __saved_models:__ models saved after training, with the best performing parameter based on validation.
* __test_images:__ images used for testing the final program.
* __dog_app.ipynb:__ a notebook that contains implemented app logic.
* __extract_bottleneck_features.py:__ a small script to extract bottleneck features suitable for each pretrained model.



## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git
cd dog-project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

__NOTE:__ While some code has already been implemented to get you started, you will need to implement additional functionality to successfully answer all of the questions included in the notebook. __Unless requested, do not modify code that has already been included.__

## Results
The implemented CNN model based on Resnet50 was able to acheive a classification accuracy of 81% for detecting dog breeds.


## Acknowledgement
This project is based on a project template provided by Udacity's Data Scientist Nanodegree program. Some of the included neural network models are based on the VGG16 and Resnet50 projects.

## More information
for more information, refer to [this blog post](https://abdelrahman-hefny.medium.com/building-a-dog-breed-classifier-using-transfer-learning-in-python-6cdd5f4d6ac8).
