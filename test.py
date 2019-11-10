import cv2, os, settings, model
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm

class TestDogCatClassifier():
    def __init__(self, model):
        self.model = model.ConvNeuralNetwork.model

    def create_testing_dataset(self):
        testing_data = []

        for img in tqdm(os.listdir(settings.TEST_DIR)):
            path = os.path.join(settings.TEST_DIR, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))

            testing_data.append([np.array(img), img_num])

        shuffle(testing_data)
        np.save('data/testing_data.npy', testing_data)

        return testing_data

    def run(self):
        self.model.load('model/{}'.format(settings.MODEL_NAME))

        if os.path.exists('data/testing_data.npy'):
            testing_data = np.load('data/testing_data.npy', allow_pickle=True)
        else:
            testing_data = self.create_testing_dataset()

        figure = plt.figure()

        for num, data in enumerate(testing_data[:20]):
            img_num = data[1]
            img_data = data[0]

            plot = figure.add_subplot(4, 5, num+1)
            orig_img = img_data
            reshaped_img = img_data.reshape(settings.IMG_SIZE, settings.IMG_SIZE, 1).astype('float32')

            model_results = self.model.predict([reshaped_img])

            if np.argmax(model_results[0]) == 1:
                label = '{}% Dog'.format(round(model_results[0][1] * 100, 2))
            else:
                label = '{}% Cat'.format(round(model_results[0][0] * 100, 2))

            plot.imshow(orig_img, cmap='gray')
            plot.axes.get_xaxis().set_visible(False)
            plot.axes.get_yaxis().set_visible(False)
            plt.title(label)

        plt.show()

if __name__ == '__main__':
    tester = TestDogCatClassifier(model)
    tester.run()
