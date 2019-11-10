import cv2, os, settings, model
import numpy as np
from random import shuffle
from tqdm import tqdm

class TrainDogCatClassifier():
    def __init__(self):
        self.model = model.ConvNeuralNetwork().model

    '''
    Conversion to one-hot array
    [1, 0] = much cat, less dog
    [0, 1] = less cat, much dog
    '''
    def label_img(self, img):
        word_label = img.split('.')[0]
        if word_label == 'cat': return [1, 0]
        if word_label == 'dog': return [0, 1]

    def create_training_dataset(self):
        training_data = []

        for img in tqdm(os.listdir(settings.TRAIN_DIR)):
            label = self.label_img(img)
            path = os.path.join(settings.TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (settings.IMG_SIZE, settings.IMG_SIZE))
            
            training_data.append([np.array(img), np.array(label)])

        shuffle(training_data)
        np.save('data/training_data.npy', training_data)

        return training_data

    def run(self):
        if os.path.exists('data/training_data.npy'):
            training_data = np.load('data/training_data.npy', allow_pickle=True)
        else:
            training_data = self.create_training_dataset()

        train_data = training_data[:-500]
        test_data = training_data[-500:]

        X = np.array([i[0] for i in train_data]).reshape(-1, settings.IMG_SIZE, settings.IMG_SIZE, 1).astype('float64')
        y = np.array([i[1] for i in train_data])

        test_X = np.array([i[0] for i in test_data]).reshape(-1, settings.IMG_SIZE, settings.IMG_SIZE, 1).astype('float64')
        test_y = np.array([i[1] for i in test_data])

        self.model.fit({'input': X}, {'targets': y}, n_epoch=20,
            validation_set=({'input': test_X}, {'targets': test_y}),
            snapshot_step=500, show_metric=True, run_id=settings.MODEL_NAME)

        self.model.save('model/{}'.format(settings.MODEL_NAME))

if __name__ == '__main__':
    trainer = TrainDogCatClassifier()
    trainer.run()
