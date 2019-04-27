import argparse
from model import get_model, freeze_all_but_mid_and_top, freeze_all_but_top
from log_cf_matrix import LogConfusionMatrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint
from dynamic_sampling import DynamicSamplingImageDataGenerator, BatchSizeAdapter


def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Sampling training on Inception-v3-based model')

    data_group = parser.add_argument_group('data')
    data_group.add_argument('num_class', type=int,
                            help='the number of classes in the dataset')
    data_group.add_argument('train_path', type=str,
                            help='path to the directory of training images')
    data_group.add_argument('valid_path', type=str,
                            help='path to the directory of validation images')
    data_group.add_argument('--img_size', nargs=2, type=int, metavar=('img_height', 'img_width'), default=(299, 299),
                            help='the target size of input images')
    data_group.add_argument('--valid_batch', type=int, default=32,
                            help='batch size during validation')
    data_group.add_argument('--batch_per_class', type=int, default=2,
                            help='batch size per class, batch_size = batch_per_class * num_class')

    augment_group = parser.add_argument_group('augment')
    augment_group.add_argument('--shear_range', type=float, default=0.2)
    augment_group.add_argument('--horizontal_flip', type=bool, default=True)
    augment_group.add_argument('--rotation_range', type=float, default=10.)
    augment_group.add_argument('--width_shift_range', type=float, default=0.2)
    augment_group.add_argument('--height_shift_range', type=float, default=0.2)

    model_group = parser.add_argument_group('model_training')
    model_group.add_argument('--weight_path', type=str, default=None,
                             help='path to the model weight file')
    model_group.add_argument('--epoch', type=int, default=1000,
                             help='the number of training epoch')
    model_group.add_argument('--log_path', type=str, default=None,
                             help='path to the log file of training process')
    model_group.add_argument('--cflog_path', type=str, default=None,
                             help='path to the log file of confusion matrix')
    model_group.add_argument('--cflog_interval', type=int, default=20,
                             help='frequency to log confusion matrix for the whole validation dataset')
    model_group.add_argument('--checkpoint_path', type=str, default=None,
                             help='path to store checkpoint model files')


    warmup_group = parser.add_argument_group('warmup')
    warmup_group.add_argument('--warmup', action='store_true',
                              help='set to train the last two layers as warmup process')
    warmup_group.add_argument('--warmup_epoch', type=int, default=50,
                              help='the number of warmup training, valid only when warmup option is used')
    args = parser.parse_args()
    return args


def main(args):
    print('Preparing Model...')
    model = get_model(args.num_class)
    if args.model_weights is not None:
        # Continue the previous training
        print('Loading saved model: \'{}\'.'.format(args.model_weights))
        model.load_weights(args.model_weights)

    print('Preparing Data...')
    datagen_train = DynamicSamplingImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)
    datagen_train = datagen_train.flow_from_directory(
        args.train_path,
        target_size=(args.img_size[0], args.img_size[1]),
        class_size_per_batch=args.class_size_per_batch)
    datagen_valid = ImageDataGenerator(rescale=1./255).flow_from_directory(
        args.valid_path,
        target_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.valid_batch)

    if args.warmup:
        print("Warm up...")
        model = freeze_all_but_top(model)
        model = model.fit_generator(
            datagen_train,
            steps_per_epoch=len(datagen_train),
            validation_data=datagen_valid,
            validation_steps=len(datagen_valid),
            epochs=args.warmup_epoch)

    print('Model Training...')
    callbacks = []
    if args.log_path is not None:
        callbacks.append(CSVLogger(args.log_path))
    if args.checkpoint_path is not None:
        callbacks.append(ModelCheckpoint(filepath=args.checkpoint_path, verbose=1, save_best_only=False))
    if args.cflog_path is not None:
        callbacks.append(LogConfusionMatrix(args.cflog_path, args.valid_path,
                                            args.cflog_interval, (args.img_size[0], args.img_size[1])))
    callbacks.append(BatchSizeAdapter((datagen_train, datagen_valid), len(datagen_valid)))

    model = freeze_all_but_mid_and_top(model)
    model = model.fit_generator(
                datagen_train,
                steps_per_epoch=len(datagen_train),
                validation_data=datagen_valid,
                validation_steps=len(datagen_valid),
                epochs=args.epoch,
                callbacks=callbacks)


if __name__ == '__main__':
    args = parse_args()
    main(args)
