from __future__ import absolute_import, division, print_function
import math, json, sys
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.datasets import cifar10, mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, PReLU, Activation
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.initializers import RandomNormal
from keras import regularizers

# designate trainable layers
def set_trainable(model, trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

# generator
def generator(input_dim):
    __z = Input(shape=(input_dim,))
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev=0.02)
    # 5th fc
    __h = Dense(units=2*2*512, activation=None, kernel_initializer=rand_init,
                kernel_regularizer=reg, bias_regularizer=reg)(__z)
    __h = Reshape((2, 2, 512))(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = Activation('relu')(__h)
    # 4th conv
    __h = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                          activation=None, kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = Activation('relu')(__h)
    # 3rd conv
    __h = Conv2DTranspose(filters=128, kernel_size=4, strides=1, padding='valid',
                          activation=None, kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = Activation('relu')(__h)
    # 2nd conv
    __h = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                          activation=None, kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = Activation('relu')(__h)
    # 1st conv
    __x = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same',
                          activation='tanh', kernel_initializer=rand_init,
                          kernel_regularizer=reg, bias_regularizer=reg)(__h)
    # return
    return Model(__z, __x, name='generator')

# discriminator
def discriminator():
    __x = Input(shape=(28, 28, 1))
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev=0.02)
    # 1st conv
    __h = Conv2D(filters=64,  kernel_size=3, strides=2, padding='same',
                 activation=None, kernel_initializer=rand_init,
                 kernel_regularizer=reg, bias_regularizer=reg)(__x)
    __h = PReLU(shared_axes=[1,2,3])(__h)
    # 2nd conv
    __h = Conv2D(filters=128, kernel_size=3, strides=2, padding='same',
                 activation=None, kernel_initializer=rand_init,
                 kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = PReLU(shared_axes=[1,2,3])(__h)
    # 3nd conv
    __h = Conv2D(filters=256, kernel_size=4, strides=1, padding='valid',
                 activation=None, kernel_initializer=rand_init,
                 kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = PReLU(shared_axes=[1,2,3])(__h)
    # 4th conv
    __h = Conv2D(filters=512, kernel_size=3, strides=2, padding='same',
                 activation=None, kernel_initializer=rand_init,
                 kernel_regularizer=reg, bias_regularizer=reg)(__h)
    __h = BatchNormalization(axis=-1)(__h)
    __h = PReLU(shared_axes=[1,2,3])(__h)
    # 5th fc
    __h = Flatten()(__h)
    __y = Dense(units=2, activation='softmax', kernel_initializer=RandomNormal(stddev=0.02),
                kernel_regularizer=reg, bias_regularizer=reg)(__h)
    # return
    return Model(__x, __y, name='discriminator')

# all the model configuration
def GAN(input_dim):
    # inputs
    __z  = Input(shape=(input_dim,))
    __xt = Input(shape=(28, 28, 1))
    # generator
    gen = generator(input_dim)  # instance of the generator function
    __xs = gen(__z)
    # discriminator
    dis = discriminator()  # instance of the discriminator function
    __yt = dis(__xt)
    __ys = dis(__xs)
    # generator training stage
    set_trainable(gen, True)
    set_trainable(dis, False)
    gen_train_stage = Model(__z, __ys, name='gen_train_stage')
    gen_train_optimizer = Adam(lr=0.0002, beta_1=0.5)
    gen_train_stage.compile(optimizer=gen_train_optimizer, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # discriminator training stage
    set_trainable(gen, False)
    set_trainable(dis, True)
    dis_train_stage = Model([__z, __xt], [__ys, __yt], name='dis_train_stage')
    dis_train_optimizer = Adam(lr=0.0002, beta_1=0.5)
    dis_train_stage.compile(optimizer=dis_train_optimizer, loss=categorical_crossentropy, metrics=[categorical_accuracy])
    # generator test stage
    set_trainable(gen, True)
    set_trainable(dis, False)
    gen_test_stage = Model(__z, __xs, name='gen_test_stage')
    gen_test_optimizer = Adam(lr=0.0002, beta_1=0.5)
    gen_test_stage.compile(optimizer=gen_test_optimizer, loss=mean_squared_error)
    # return
    return gen_train_stage, dis_train_stage, gen_test_stage

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
    input_dim = 100
    max_epoch = 10000
    batch_size = 100

    # load the MNIST dataset
    print('Loading the dataset...')
#    (X_train_orig, _), (X_test_orig, _) = cifar10.load_data()
    (X_train_orig, _), (X_test_orig, _) = mnist.load_data()
#    X_train = (X_train_orig.astype(np.float32) - 128.0) / 128.0
#    X_test  = (X_test_orig .astype(np.float32) - 128.0) / 128.0
    X_train = (X_train_orig.astype(np.float32)[:,:,:,np.newaxis] - 128.0) / 128.0
    X_test  = (X_test_orig .astype(np.float32)[:,:,:,np.newaxis] - 128.0) / 128.0
    n_samples_train = X_train.shape[0]
    n_samples_test  = X_test.shape[0]

    # model definitions
    print('Model definitions...')
    gen_train_stage, dis_train_stage, gen_test_stage = GAN(input_dim)
    plot_model(gen_train_stage.layers[1], to_file='model.png', show_shapes=True, show_layer_names=True)

    # for each epoch
    history = dict()
    print('Training...')
    for epoch in tqdm(range(max_epoch)):
        # random permutation
        ns = np.random.permutation(n_samples_train)
        # for each batch
        gen_loss = 0
        gen_acc = 0
        dis_loss = 0
        dis_acc = 0
        num_batches = int(math.floor(n_samples_train / batch_size))
        for batch in tqdm(range(num_batches)):
            # batch size
            batch_start = batch * batch_size
            batch_end = min((batch+1)*batch_size, n_samples_train)
            now_batch_size = batch_end - batch_start
            # batch of images
            X_train_batch = X_train[ns][batch_start:batch_end]
            # batch of random vectors
            Z_train_batch = np.asarray(np.random.uniform(-1.0, 1.0, size=(now_batch_size, input_dim)), dtype=np.float32)
            # batch of discriminator labels
            Y_true_batch = to_categorical(np.ones(now_batch_size),  num_classes=2)
            Y_fake_batch = to_categorical(np.zeros(now_batch_size), num_classes=2)
            # generator training stage
            gen_loss_now, gen_acc_now = gen_train_stage.train_on_batch(Z_train_batch, Y_true_batch)
            gen_loss += gen_loss_now
            gen_acc  += gen_acc_now
            # discriminator training stage
            dis_loss_now, dis_fake_loss, dis_true_loss, dis_fake_acc, dis_true_acc = \
                dis_train_stage.train_on_batch([Z_train_batch, X_train_batch], [Y_fake_batch, Y_true_batch])
            dis_acc_now = (dis_fake_acc  + dis_true_acc) / 2
            dis_loss += dis_loss_now
            dis_acc  += dis_acc_now
            print('\r gen_loss = {0:.3f}, dis_loss = {1:.3f}, gen_acc = {2:.3f}, dis_acc_t = {3:.3f}, dis_acc_f = {4:.3f}'
                  .format(gen_loss_now, dis_loss_now, gen_acc_now, dis_true_acc, dis_fake_acc), end='')
            sys.stdout.flush()
        # end for batch in tqdm(range(num_batches))

        # loss and accuracy
        gen_loss /= num_batches
        dis_loss /= num_batches
        print('Generator training loss:         {0}'.format(gen_loss))
        print('Discriminator training loss:     {0}'.format(dis_loss))
        gen_acc /= num_batches
        dis_acc /= num_batches
        print('Generator training accuracy:     {0}'.format(gen_acc))
        print('Discriminator training accuracy: {0}'.format(dis_acc))

        # test
        Z_test = np.random.uniform(-1.0, 1.0, size=(n_samples_test, input_dim))
        Y_test_true = to_categorical(np.ones(n_samples_test),  num_classes=2)
        Y_test_fake = to_categorical(np.zeros(n_samples_test), num_classes=2)
        test_loss, test_true_loss, test_fake_loss, test_true_acc, test_fake_acc = \
            dis_train_stage.evaluate([Z_test, X_test], [Y_test_fake, Y_test_true], batch_size=batch_size, verbose=1)
        test_acc  = (test_true_acc  + test_fake_acc)  / 2
        print('Test loss:     {0}'.format(test_loss))
        print('Test accuracy: {0}'.format(test_acc))

        # training history
        history[epoch] = {
            'gen_train_loss': gen_loss, 'dis_train_loss': dis_loss, 'test_loss': test_loss,
            'gen_train_acc' : gen_acc,  'dis_train_acc' : dis_acc,  'test_acc' : test_acc,
        }

        # generate
        if epoch % 1 == 0:
            sim_x = 10
            sim_y = 10
            n_samples_sim = sim_x * sim_y
            X_sim = gen_test_stage.predict(Z_test[:n_samples_sim]) * 128.0 + 128.0
            # display
            for n in range(n_samples_sim):
                plt.subplot(sim_x, sim_y, n+1)
#                plt.imshow(X_sim[n])
                plt.imshow(X_sim[n].squeeze())
                plt.gray()
                plt.tick_params(labelbottom="off", bottom="off", labelleft="off",left="off")
            seed_zfill = str(epoch).zfill(5)
            plt.savefig('generated_images_{0}.png'.format(seed_zfill))

    # end for epoch in tqdm(range(max_epoch))

    # save the results
    ## model definitions
    with open('modeldefs.json', 'w') as fout:
        model_json_str = gen_train_stage.to_json(indent=4)
        fout.write(model_json_str)
    ## model weights
    gen_train_stage.save_weights('modelweights.hdf5')
    ## training history
    json.dump(history, open('training_history.json', 'w'))
