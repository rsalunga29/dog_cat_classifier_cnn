# Directories
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Network settings
IMG_SIZE = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.8

# Model settings
MODEL_NAME = 'dogs_vs_cats_{}_{}.tfl'.format(str(LEARNING_RATE).replace('.', '-'), '4conv_20e-augmented-1')
