import tensorflow as tf
from tensorflow import keras

filters = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
kernels = [7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
strides = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]


def resBlock(x, out_channel, strides, name="unit"):
    shortcut = []
    is_training = tf.get_variable('is_training', (), dtype=tf.bool, trainable=False)
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = tf.layers.conv2d(
                                        inputs=x,
                                        filters=out_channel,
                                        kernel_size=1,
                                        strides=(strides, strides),
                                        padding="same",
                                        kernel_regularizer=keras.regularizers.l2(0.005),
                                        activation=None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name='shortcut')
            shortcut = tf.contrib.layers.batch_norm(inputs=shortcut, is_training=is_training, scope='bn_shortcut')

        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=out_channel,
            kernel_size=3,
            strides=(strides, strides),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv1')
        bn1 = tf.layers.batch_normalization(
            inputs=conv1,
            training=is_training
            )
        relu1 = tf.nn.relu(features=bn1, name='relu1')
        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=out_channel,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv2')
        bn2 = tf.layers.batch_normalization(
            inputs=conv2,
            training=is_training
            )
        relu2 = tf.nn.relu(features=bn2, name='relu2')
        y = relu2 + shortcut
        #y = tf.nn.relu(features=bn2, name='relu2')

        return y


def create(x, num_outputs):
    '''
        args:
            x               network input
            num_outputs     number of logits
    '''
    
    is_training = tf.get_variable('is_training', (), dtype = tf.bool, trainable = False)
    with tf.name_scope('resnet'):

        conv0 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=7,
            strides=(2, 2),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv0")

        bn0 = tf.layers.batch_normalization(
            inputs=conv0,
            training=is_training
            )
        relu0 = tf.nn.relu(features=bn0, name='relu0')
        pool0 = tf.layers.max_pooling2d(inputs=relu0, pool_size=(3, 3), strides=(2, 2), padding="same", name='pool0')
        block1 = resBlock(x=pool0, out_channel=filters[1], strides=strides[1], name="block1")
        block2 = resBlock(x=block1, out_channel=filters[3], strides=strides[3], name="block2")
        block3 = resBlock(x=block2, out_channel=filters[5], strides=strides[5], name="block3")
        block4 = resBlock(x=block3, out_channel=filters[7], strides=strides[7], name="block4")
        block5 = resBlock(x=block4, out_channel=filters[9], strides=strides[9], name="block5")
        block6 = resBlock(x=block5, out_channel=filters[11], strides=strides[11], name="block6")
        block7 = resBlock(x=block6, out_channel=filters[13], strides=strides[13], name="block7")
        block8 = resBlock(x=block7, out_channel=filters[15], strides=strides[15], name="block8")
        avg_pool = tf.reduce_mean(block8, [1, 2])

        output = tf.contrib.layers.fully_connected(inputs=avg_pool, num_outputs=num_outputs, activation_fn=None, scope="fc")

    return output

    # TODO


