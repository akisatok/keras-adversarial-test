from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn import metrics
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.backend import tensorflow_backend
from keras.utils.np_utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

class Adversarial_Training(Model):
    _at_loss = None
    # set up
    def setup_at_loss(self, loss_func=categorical_crossentropy, eps=0.25/255.0, alpha=1.0):
        self._loss_func = loss_func
        self._alpha = alpha
        self._at_loss = self.at_loss(eps)
        return self
    # loss
    @property
    def losses(self):
        losses = super(self.__class__, self).losses
        if self._at_loss is not None:
            losses += [ self._alpha * self._at_loss ]
        return losses
    # VAT loss
    def at_loss(self, eps):
        # original loss
        loss_orig = self._loss_func(self.inputs[1], self.outputs[0])
        # gradients
        grads = K.stop_gradient(K.gradients(loss_orig, self.inputs[0]))[0]
        # perterbed samples
        new_inputs = self.inputs[0] + eps * K.sign(grads)
        # estimation for the perturbated samples
        outputs_perturb = self.call([new_inputs, self.inputs[1]])
        # computing losses
        loss = self._loss_func(self.inputs[1], outputs_perturb)
        return loss

def CNN_Model_with_AT(input_shape, output_dim, loss_func=categorical_crossentropy, eps=0.25, alpha=0.5):
    # core model
    __x = Input(shape=input_shape)
    __h = Conv2D(filters=32, kernel_size=3, activation='relu')(__x)
    __h = Conv2D(filters=64, kernel_size=3, activation='relu')(__h)
    __h = MaxPooling2D(pool_size=(2, 2))(__h)
    __h = Dropout(rate=0.25)(__h)
    __h = Flatten()(__h)
    __h = Dense(units=128, activation='relu')(__h)
    __h = Dropout(rate=0.25)(__h)
    __yp = Dense(units=output_dim, activation='softmax')(__h)
    # adversarial training
    __yt = Input(shape=(output_dim,))
    # return
    return Adversarial_Training([__x, __yt], __yp).setup_at_loss(loss_func=loss_func, eps=eps, alpha=alpha)

# main
if __name__ == "__main__":
    # GPU configulations
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)

    # random seeds
    np.random.seed(1)
    tf.set_random_seed(1)

    # parameters
    n_classes = 10
    n_channels = 1
    img_width = 28
    img_height = 28

    # load the dataset
    print('Loading the dataset...')
    from keras.datasets import mnist as dataset
    (X_train, Y_train_int), (X_test, Y_test_int) = dataset.load_data()
    X_train = X_train[:, np.newaxis].transpose((0, 2, 3, 1)).astype('float32') / 255.0
    X_test = X_test[:, np.newaxis].transpose((0, 2, 3, 1)).astype('float32') / 255.0
    Y_train = to_categorical(Y_train_int, num_classes=n_classes)
    Y_test = to_categorical(Y_test_int, num_classes=n_classes)

    # training
    print('Train a NN model...')
    ## define
    input_shape = (img_width, img_height, n_channels)
    model = CNN_Model_with_AT(input_shape, n_classes, loss_func=categorical_crossentropy)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    ## train
    history = model.fit([X_train, Y_train], Y_train, batch_size=100,
                        epochs=2, shuffle=True, initial_epoch=0)

    # test
    Y_train_pred = model.predict([X_train, Y_train])
    Y_train_pred = Y_train_pred.argmax(axis=1)
    Y_test_pred  = model.predict([X_test, Y_test])
    Y_test_pred = Y_test_pred.argmax(axis=1)
    print('Training score for a NN classifier: \t{0}'.format(
        metrics.accuracy_score(Y_train_int, Y_train_pred)))
    print('Test score for a NN classifier: \t{0}'.format(
        metrics.accuracy_score(Y_test_int, Y_test_pred)))
    print('Training classification report for a NN classifier\n{0}\n'.format(
        metrics.classification_report(Y_train_int, Y_train_pred)))
    print('Test classification report for a NN classifier\n{0}\n'.format(
        metrics.classification_report(Y_test_int, Y_test_pred)))
