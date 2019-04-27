from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD


def get_model(num_class, hidden_dim=1024, activation='relu', weights='imagenet'):
    """
    Instantiates the Inception-v3-based model

    :param num_class: the number of classes in the dataset
    :param hidden_dim: the dimension of last hidden layer
    :param activation: the activation function used in the last hidden layer
    :param weights: one of `None` (random initialization),
                `imagenet` (pre-training on ImageNet),
                or the path to the weights file to be loaded.
    :return: A Keras model instance
    """

    # Define model structure
    base_model = InceptionV3(weights=weights, include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(hidden_dim, activation=activation)(x)
    pred = Dense(num_class, activation='softmax')(x)

    # Build model
    return Model(inputs=base_model.input, outputs=pred)


def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model