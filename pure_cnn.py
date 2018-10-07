import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator

from helper import get_class_names, get_train_data, get_test_data, plot_images
from helper import plot_model, predict_classes, visualize_errors
matplotlib.style.use('ggplot')

class_names = get_class_names()
print(class_names)

num_classes = len(class_names)
print(num_classes)


# Hight and width of the images# Hight
IMAGE_SIZE = 32
# 3 channels, Red, Green and Blue
CHANNELS = 3
# Number of epochs
NUM_EPOCH = 350
# learning rate
LEARN_RATE = 1.0e-4

images_train, labels_train, class_train = get_train_data()
images_test, labels_test, class_test = get_test_data()

print("Training set size:\t",len(images_train))
print("Testing set size:\t",len(images_test))

"""Different from the simple vision,
this time we use the pure cnn (all conv layer)"""


def pure_cnn_model():
    model = Sequential()

    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    model.add(GlobalAveragePooling2D())

    model.add(Activation('softmax'))

    model.summary()

    return model

    model = pure_cnn_model()
    checkpoint = ModelCheckpoint('best_model_improved.h5',  # model filename
                                 monitor='val_loss',  # quantity to monitor
                                 verbose=0,  # verbosity - 0 or 1
                                 save_best_only=True,  # The latest best model will not be overwritten
                                 mode='auto')  # The decision to overwrite model is made
    # automatically depending on the quantity to monitor
    model.compile(loss='categorical_crossentropy',  # Better loss function for neural networks
                  optimizer=Adam(lr=LEARN_RATE),  # Adam optimizer with 1.0e-4 learning rate
                  metrics=['accuracy'])  # Metrics to be evaluated by the model

    model_details = model.fit(images_train, class_train,
                              batch_size=128,
                              epochs=NUM_EPOCH,  # number of iterations
                              validation_data=(images_test, class_test),
                              callbacks=[checkpoint],
                              verbose=1)

    scores = model.evaluate(images_test, class_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    plot_model(model_details)



    #do the augmentation before set up the model

    datagen == ImageDataGeneratorImageDat(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(images_train)
    augmented_model = pure_cnn_model()
    augmented_checkpoint = ModelCheckpoint('augmented_best_model.h5',  # model filename
                                           monitor='val_loss',  # quantity to monitor
                                           verbose=0,  # verbosity - 0 or 1
                                           save_best_only=True,  # The latest best model will not be overwritten
                                           mode='auto')  # The decision to overwrite model is made
    # automatically depending on the quantity to monitor
    augmented_model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model

    augmented_model_detailsaugmente = augmented_model.fit_generator(
        datagen.flow(images_train, class_train, batch_size=32),
        steps_per_epoch=len(images_train) / 32,  # number of samples per gradient update
        epochs=NUM_EPOCH,  # number of iterations
        validation_data=(images_test, class_test),
        callbacks=[augmented_checkpoint],
        verbose=1)