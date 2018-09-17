import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.models import load_model

from helper import get_class_names, get_train_data, get_test_data, plot_images, plot_model

images_test, labels_test, class_test = get_test_data()
class_name = get_class_names()
model = load_model("best_model_simple.h5")
pred = model.predict(images_test, batch_size = 32)
# print(pred[0])

exp_label = np.argmax(pred,axis=1)
# print(exp_label)

correct = (labels_test == exp_label)
incorrect = (labels_test != exp_label)

mis_images = images_test[incorrect]
mis_labels = exp_label[incorrect]
correct_labels = labels_test[incorrect]

plot_images(images=images_test[0:16],
            labels_true=labels_test[0:16],
            class_names=class_name,
            labels_pred=exp_label[0:16])
# print(sum(correct)/len(images_test))