import cv2, os, argparse, settings, model
import numpy as np
import matplotlib.pyplot as plt

class DogCatClassifier():
    def __init__(self, args, model):
        self.model = model.ConvNeuralNetwork.model
        self.args = args

    def run(self):
        self.model.load('model/{}'.format(settings.MODEL_NAME))

        # Will be used as display image for plt
        orig_img = cv2.imread(self.args.image)
        
        # Get b,g,r to convert to r,g,b
        b,g,r = cv2.split(orig_img)
        orig_img = cv2.merge([r,g,b])

        cv2_img = cv2.imread(self.args.image, cv2.IMREAD_GRAYSCALE)
        cv2_img = cv2.resize(cv2_img, (settings.IMG_SIZE, settings.IMG_SIZE))

        img_data = np.array(cv2_img)
        reshaped_img = img_data.reshape(settings.IMG_SIZE, settings.IMG_SIZE, 1).astype('float32')

        # Get prediction
        prediction = self.model.predict([reshaped_img])

        print(prediction)

        if np.argmax(prediction[0]) == 1:
            label = '{}% Dog'.format(round(prediction[0][1] * 100, 2))
        else:
            label = '{}% Cat'.format(round(prediction[0][0] * 100, 2))

        plt.figure(num='Image Classifier')
        plt.imshow(orig_img)
        plt.title(label)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Image Classifier",
        description="Dog vs Cat Classifier using CNNs by github.com/rsalunga29")
    parser.add_argument('-i', '--image', type=str, required=True, help="Absolute path to image that needs to be classified.")
    args = parser.parse_args()

    classifier = DogCatClassifier(args, model)
    classifier.run()
