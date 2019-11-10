import tflearn, settings
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class ConvNeuralNetwork():
    # Image preprocessing
    imgprep = tflearn.ImagePreprocessing()
    imgprep.add_featurewise_zero_center()
    imgprep.add_featurewise_stdnorm()

    # Image augmentation
    imgaug = tflearn.ImageAugmentation()
    imgaug.add_random_rotation()
    imgaug.add_random_flip_leftright()

    # Input layer
    convnet = input_data(shape=[None, settings.IMG_SIZE, settings.IMG_SIZE, 1],
        data_preprocessing=imgprep, data_augmentation=imgaug, name='input')

    # Hidden layers
    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 3, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    # Fully connected layer
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, settings.DROPOUT_RATE)

    # Output layer
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy', learning_rate=settings.LEARNING_RATE, name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
