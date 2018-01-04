from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy import random
from numpy.linalg import norm

from keras.engine.topology import Container
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical

def set_trainable(model, trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

def test_model_l0():
    __x = Input(shape=(5,))
    __h = test_model_l1(name='model_l1_1')(__x)
    __h = test_model_l1(name='model_l1_2')(__h)
    __y = Dense(units=2, activation='softmax')(__h)
    return Model(__x, __y, name='model_l0')

def test_model_l1(name):
    __x = Input(shape=(5,))
    __h = test_model_l2(name='model_l2_1')(__x)
    __h = test_model_l2(name='model_l2_2')(__h)
    return Container(__x, __h, name=name)

def test_model_l2(name):
    __x = Input(shape=(5,))
    __h = test_model_l3(name='model_l3_1')(__x)
    __h = test_model_l3(name='model_l3_2')(__h)
    return Container(__x, __h, name=name)

def test_model_l3(name):
    __x = Input(shape=(5,))
    __h = Dense(units=5, activation='relu')(__x)
    __h = BatchNormalization(axis=-1)(__h)
    return Container(__x, __h, name=name)

if __name__ == "__main__":
    # model definitions
    test_model = test_model_l0()
    optimizer = Adam()
    test_model.compile(optimizer=optimizer, loss=categorical_crossentropy)
    test_model.summary()
    test_model_weights = test_model.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]

    # set trainable on the top
    test_model_2 = test_model_l0()
    test_model_2.trainable = False
    optimizer_2 = Adam()
    test_model_2.compile(optimizer=optimizer_2, loss=categorical_crossentropy)
    test_model_2.summary()
    test_model_2_weights = test_model_2.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]
    print(test_model_2.layers[1].trainable)
    print(test_model_2.layers[1].layers[1].trainable)

    # set trainable recursively
    test_model_3 = test_model_l0()
    set_trainable(test_model_3, False)
    optimizer_3 = Adam()
    test_model_3.compile(optimizer=optimizer_3, loss=categorical_crossentropy)
    test_model_3.summary()
    test_model_3_weights = test_model_3.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]
    print(test_model_3.layers[1].trainable)
    print(test_model_3.layers[1].layers[1].trainable)

    # train all the models with random vectors
    X = random.normal(size=(1000,5))
    Y = to_categorical(random.random_integers(0, 1, size=1000))
    test_model.fit(X, Y, batch_size=100, epochs=1, verbose=0)
    test_model_2.fit(X, Y, batch_size=100, epochs=1, verbose=0)
    test_model_3.fit(X, Y, batch_size=100, epochs=1, verbose=0)

    # check
    test_model_weights_new = test_model.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]
    test_model_2_weights_new = test_model_2.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]
    test_model_3_weights_new = test_model_3.get_layer(name='model_l1_1').get_layer(name='model_l2_1').get_layer(name='model_l3_1').layers[1].get_weights()[0]
    print(norm(test_model_weights_new - test_model_weights))
    print(norm(test_model_2_weights_new - test_model_2_weights))
    print(norm(test_model_3_weights_new - test_model_3_weights))

