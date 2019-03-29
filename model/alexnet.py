import tensorflow as tf
from tensorflow import keras


def create(x, num_outputs, dropout):
    '''
        args:
            x               network input
            num_outputs     number of logits
            dropout         dropout rate during training
    '''
    is_training = tf.get_variable('is_training', (), dtype=tf.bool, trainable=False)
    
    with tf.name_scope('alexnet'):
        
        # TODO
        # Weights for 2-D convolution with 3x3 kernel , from 3 input to
        # 16 output channels , initialized with Xavier
        dropout_rate=dropout

        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=96,
            kernel_size=11,
            strides=(4, 4),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv1")


        bn1 = tf.layers.batch_normalization(
            inputs=conv1,
            training=is_training
           )

        relu1 = tf.nn.relu(features=bn1, name='relu1')

        pool1 = tf.layers.max_pooling2d(
            inputs=relu1,
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid")

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=192,
            kernel_size=5,
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv2")

        bn2 = tf.layers.batch_normalization(
            inputs=conv2,
            training=is_training
            )

        relu2 = tf.nn.relu(features=bn2, name='relu2')

        pool2 = tf.layers.max_pooling2d(
            inputs=relu2,
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid")

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=384,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv3")

        bn3 = tf.layers.batch_normalization(
            inputs=conv3,
            training=is_training
           )

        relu3 = tf.nn.relu(features=bn3, name='relu3')

        conv4 = tf.layers.conv2d(
            inputs=relu3,
            filters=256,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv4")

        bn4 = tf.layers.batch_normalization(
            inputs=conv4,
            training=is_training
            )

        relu4 = tf.nn.relu(features=bn4, name='relu4')

        conv5 = tf.layers.conv2d(
            inputs=relu4,
            filters=256,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            kernel_regularizer=keras.regularizers.l2(0.005),
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="conv5")

        bn5 = tf.layers.batch_normalization(
            inputs=conv5,
            training=is_training
            )

        relu5 = tf.nn.relu(features=bn5, name='relu5')

        pool3 = tf.layers.max_pooling2d(
            inputs=relu5,
            pool_size=(3, 3),
            strides=(2, 2),
            padding="valid")

        dropout1 = tf.layers.dropout(
            inputs=pool3,
            rate=dropout_rate,
            training=is_training
        )
        
        flatten = tf.layers.flatten(dropout1)

        
       
        
        fc1 = tf.contrib.layers.fully_connected(
            inputs=flatten,
            num_outputs=4096,
            weights_regularizer=keras.regularizers.l2(0.005),
            scope="fc1"
        )
       
        dropout2 = tf.layers.dropout(
            inputs=fc1,
            rate=dropout_rate,
            training=is_training
        )
        
        fc2 = tf.contrib.layers.fully_connected(
            inputs=dropout2,
            num_outputs=4096,
            weights_regularizer=keras.regularizers.l2(0.005),
            scope="fc2"
        )
        #relu6 = tf.nn.relu(features=fc2, name='relu6')

        output = tf.contrib.layers.fully_connected(
            inputs=fc2,
            num_outputs=num_outputs,
            weights_regularizer=keras.regularizers.l2(0.005),
            scope="fc3",
            activation_fn=None
        )

    return output






