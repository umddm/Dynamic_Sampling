import numpy as np
import math
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


class LogConfusionMatrix(keras.callbacks.Callback):
    def __init__(self, log_path, image_directory, interval, image_size, on_screen=True):
        self.log_path = log_path
        image_gen = ImageDataGenerator(rescale=1./255)
        self.gen = image_gen.flow_from_directory(image_directory, image_size,
                                                 batch_size=32, shuffle=False, class_mode='sparse')
        self.image_directory = image_directory
        self.interval = interval
        self.on_screen = on_screen

        self.logger = open(self.log_path, 'w')
        self.classes = None
        super(LogConfusionMatrix, self).__init__()

    def on_train_begin(self, logs=None):
        if self.logger.closed:
            self.logger = open(self.log_path, 'a')

    def on_train_end(self, logs=None):
        self.logger.close()

    def on_epoch_end(self, epoch, logs=None):
        # Print Confusion Matrix every #interval epoch
        if epoch % self.interval == 0:
            true_y = np.array([])
            pred_y = np.array([])
            for _ in range(self.gen.samples // self.gen.batch_size):
                # Get input data
                batch_x, batch_y = next(self.gen)
                # Generate prediction on data
                batch_pred_y = self.model.predict(batch_x)
                true_y = np.append(true_y, batch_y)
                pred_y = np.append(pred_y, np.argmax(batch_pred_y, axis=1))
            if self.gen.samples % self.gen.batch_size != 0:
                batch_x, batch_y = next(self.gen)
                batch_pred_y = self.model.predict(batch_x)
                true_y = np.append(true_y, batch_y)
                pred_y = np.append(pred_y, np.argmax(batch_pred_y, axis=1))
            # Generate confusion matrix
            cf = confusion_matrix(true_y, pred_y)
            precision = np.array([])
            recall = np.array([])
            for i in range(len(cf)):
                if np.sum(cf[i]) == 0:
                    precision = np.append(precision, 1)
                else:
                    precision = np.append(precision, cf[i][i] / np.sum(cf[i]))
                if np.sum(cf, axis=0)[i] == 0:
                    recall = np.append(recall, 0)
                else:
                    recall = np.append(recall, cf[i][i] / np.sum(cf, axis=0)[i])
            f1 = np.zeros((len(cf)))
            for i in range(len(cf)):
                if precision[i] + recall[i] == 0:
                    f1[i] = 0
                else:
                    f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

            # Generate output
            self.logger.write("Epoch {}:\n".format(epoch))
            self.logger.write(np.array2string(cf, 120))
            self.logger.write("\n----------------------------------------------\n")
            self.logger.write("{:15}{}\n".format("Class:", np.array2string(np.array(list(self.gen.class_indices.keys())), 120,
                                                                         formatter={'str_kind': '{:^7.7}'.format})))
            self.logger.write("{:15}{}\n".format("Precision:",
                                               np.array2string(precision, 120, formatter={'float_kind': '{:^.7}'.format})))
            self.logger.write("{:15}{}\n".format("Recall:",
                                               np.array2string(recall, 120, formatter={'float_kind': '{:^.7}'.format})))
            self.logger.write("{:15}{}\n".format("F1-Score:",
                                               np.array2string(f1, 120, formatter={'float_kind': '{:^.7}'.format})))
            self.logger.write("{:15}{}\n".format("Ave Pre:", np.mean(precision)))
            self.logger.write("{:15}{}\n".format("Ave Recall:", np.mean(recall)))
            self.logger.write("{:15}{}\n".format("Ave F1:", np.mean(f1)))

            if self.on_screen:
                print("Epoch {}:".format(epoch))
                print(np.array2string(cf, 120))
                print("----------------------------------------------")
                print("{:15}{}".format("Precision:",
                                               np.array2string(precision,120,formatter={'float_kind': '{:^.7}'.format})))
                print("{:15}{}".format("Recall:",
                                               np.array2string(recall,120,formatter={'float_kind': '{:^.7}'.format})))
                print("{:15}{}".format("F1-Score:",
                                               np.array2string(f1,120,formatter={'float_kind': '{:^.7}'.format})))
                print("{:15}{}\n".format("Ave Pre:", np.mean(precision)))
                print("{:15}{}\n".format("Ave Recall:", np.mean(recall)))
                print("{:15}{}\n".format("Ave F1:", np.mean(f1)))
