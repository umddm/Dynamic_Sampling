import os
import threading
import numpy as np
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import keras.callbacks
from sklearn.metrics import confusion_matrix


class DynamicSamplingImageDataGenerator(ImageDataGenerator):
    """
    Data Generator for Dynamic Sampling based on keras.preprocessing.image.ImageDataGenerator
    """

    def __init__(self, *args, **kwargs):
        super(DynamicSamplingImageDataGenerator, self).__init__(*args, **kwargs)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):
        raise NotImplementedError('flow function has not been implemented for DynamicSamplingImageDataGenerator')

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            class_size_per_batch=None, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        if class_size_per_batch is None:
            raise ValueError('class_size_per_batch is necessary for DynamicSamplingImageDataGenerator')

        return ClassNumControlledDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            class_size_per_batch=class_size_per_batch, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)


class ClassNumControlledDirectoryIterator(object):
    """
    Directory Iterator for Dynamic Sampling based on keras.preprocessing.image.DirectoryIterator
    """
    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 class_size_per_batch=None, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.samples = dict()
        self.total_samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        for _class in classes:
            self.samples[self.class_indices[_class]] = 0

        def _recursive_list(subpath):
            return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        self.samples[self.class_indices[subdir]] += 1
                        self.total_samples += 1
        print('Found %d images belonging to %d classes.' % (self.total_samples, self.num_class))

        if not class_size_per_batch:
            class_size_per_batch = self.total_samples / len(classes)
        self.class_size_per_batch = int(class_size_per_batch)

        # second, build an index of the images in the different class subfolders
        self.filenames = dict()

        for subdir in classes:
            self.filenames[self.class_indices[subdir]] = []
            subpath = os.path.join(directory, subdir)
            for root, _, files in _recursive_list(subpath):
                for fname in files:
                    is_valid = False
                    for extension in white_list_formats:
                        if fname.lower().endswith('.' + extension):
                            is_valid = True
                            break
                    if is_valid:
                        # add filename relative to directory
                        absolute_path = os.path.join(root, fname)
                        self.filenames[self.class_indices[subdir]].append(os.path.relpath(absolute_path, directory))

        self.shuffle = shuffle
        self.classes = classes
        self.batch_index = dict()
        self.total_batches_seen = dict()
        self.index_generator = dict()
        self.batch_size = dict()
        for _class in classes:
            self.batch_index[self.class_indices[_class]] = 0
            self.total_batches_seen[self.class_indices[_class]] = 0
            self.index_generator[self.class_indices[_class]] = \
                self._flow_index(_class, seed)
            self.batch_size[self.class_indices[_class]] = self.class_size_per_batch

            while len(self.filenames[self.class_indices[_class]]) <= self.batch_size[self.class_indices[_class]]:
                self.filenames[self.class_indices[_class]].extend(self.filenames[self.class_indices[_class]])

        self.lock = threading.Lock()

    def update_size(self, weight):
        if sum(weight) == 0:
            for i in range(len(self.classes)):
                self.batch_size[i] = self.class_size_per_batch
        else:
            for i in range(len(self.classes)):
                weight = weight / sum(weight)
                self.batch_size[i] = int(weight[i] * self.class_size_per_batch * len(self.classes))
                while len(self.filenames[i]) <= self.batch_size[i]:
                    self.filenames[i].extend(self.filenames[i])

    def print_size(self):
        print(self.batch_size)

    def reset(self, _class):
        self.batch_index[self.class_indices[_class]] = 0

    def _flow_index(self, _class, seed=None):
        # Ensure self.batch_index[self.class_indices[_class]] is 0.
        self.reset(_class)
        n = len(self.filenames[self.class_indices[_class]])
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen[self.class_indices[_class]])
            if self.batch_index[self.class_indices[_class]] == 0:
                index_array = np.arange(n)
                if self.shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index[self.class_indices[_class]] *
                             self.batch_size[self.class_indices[_class]]) % n
            if n > current_index + self.batch_size[self.class_indices[_class]]:
                current_batch_size = self.batch_size[self.class_indices[_class]]
                self.batch_index[self.class_indices[_class]] += 1
            else:
                current_batch_size = n - current_index
                self.batch_index[self.class_indices[_class]] = 0
            self.total_batches_seen[self.class_indices[_class]] += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        index_array = dict()
        current_index = dict()
        current_batch_size = dict()
        total_size = 0
        with self.lock:
            for _, _class in self.class_indices.items():
                index_array[_class], current_index[_class], current_batch_size[_class] = next(self.index_generator[_class])
                total_size += current_batch_size[_class]
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((total_size,) + self.image_shape, dtype=K.floatx())
        batch_y = np.zeros(total_size, dtype=K.floatx())
        offset = 0
        for _, _class in self.class_indices.items():
            grayscale = self.color_mode == 'grayscale'
            # build batch of image data for each class
            for i, j in enumerate(index_array[_class]):
                fname = self.filenames[_class][j]
                img = load_img(os.path.join(self.directory, fname),
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i + offset] = x
                batch_y[i + offset] = _class
            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i in range(current_batch_size[_class]):
                    img = array_to_img(batch_x[i+offset], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=_class,
                                                                      index=current_index[_class] + i,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            offset += current_batch_size[_class]

        # build batch of labels
        if self.class_mode == 'sparse':
            pass
        elif self.class_mode == 'binary':
            batch_y = batch_y.astype(K.floatx())
        elif self.class_mode == 'categorical':
            label = batch_y
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i in range(len(label)):
                batch_y[i, int(label[i])] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class BatchSizeAdapter(keras.callbacks.Callback):
    def __init__(self, generators, step):
        self.train_generator, self.validation_generator = generators
        self.step = step
        self.classes = self.train_generator.classes
        self.num_class = len(self.train_generator.classes)
        super(BatchSizeAdapter, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.train_generator.print_size()

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.array([])
        y_pred = np.array([])
        for _ in range(self.step):
            batch_x, batch_y = next(self.validation_generator)
            batch_y_pred = self.model.predict(batch_x)
            for i in range(len(batch_x)):
                y_true = np.append(y_true, np.argmax(batch_y[i]))
                y_pred = np.append(y_pred, np.argmax(batch_y_pred[i]))

        cf = confusion_matrix(y_true, y_pred, labels=np.array(range(self.num_class)))

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
            if precision[i]+recall[i] == 0:
              f1[i] = 0
            else:
              f1[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])

        self.train_generator.update_size(1 - f1)

