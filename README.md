# Image Classifier with Deep Learning
<h3 align="center">
    <img src="https://images2.minutemediacdn.com/image/upload/c_crop,h_706,w_1256,x_0,y_64/f_auto,q_auto,w_1100/v1554995050/shape/mentalfloss/516438-istock-637689912.jpg)
Image from [Mentalfloss.com](https://www.mentalfloss.com/article/516438/8-tricks-help-your-cat-and-dog-get-along" width="300">
    Image credit from [Mentalfloss.com](https://www.mentalfloss.com/article/516438/8-tricks-help-your-cat-and-dog-get-along)
</h3>

### Introduction
This is my 🐶 vs 🐱 image classifier with **Convolutional Neural Networks (CNNs) using Tensorflow 1.x and TFLearn**. I have created this after following a couple of tutorials from [Sentdex](https://pythonprogramming.net/). This is merely a test and practice, I do not own the dataset that is used and most of my codebase is heavily influence by Sentdex's tutorial.

### Usage
1.) Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition#)
2.) Extract the files on the data folder
3.) After extracting, go inside the data folder and extract the train and test into their respective folders
4.) Tweak the network and model settings by opening `settings.py` and change the number of epochs *(n_epoch)* by opening `train.py` and editing *line 51*
5.) Run `python3 train.py` on your terminal to train the model
6.) Wait for the training to be finished, run `tensorboard --dirlog=log` on your terminal to see the performance of the model
7.) Run `python3 test.py` to test your model's accuracy
8.) Run `python3 main.py --image="some/path/to/image.jpg"` to predict whether your image is a 🐶 or a 🐱
 
### Tweaking network and model
By default, the `LEARNING_RATE` is set to `1e-4` or `0.0001` while the `DROPOUT_RATE` is set to `0.8`, you can change this by opening `settings.py` and editing their values.

For the model that I'm using, I already implemented `Image PreProcessing` and `Image Augmentation`, in order to add or remove preprocessing functions or augmentation functions, checkout [TFLearn's documentation](http://tflearn.org/).

Additionally, the model is structured to be have 4 hidden layers, you can change this by opening `model.py` and add or remove hidden layers to your desired quantity.

To change the number of epochs*, open `train.py` and edit the `n_epoch` parameter on *line 51*

*an epoch refers to one cycle through the full training dataset.
