from argparse import ArgumentParser
from pathlib import Path
import os
from matplotlib import pyplot as plt
import save_helper as sh
import tensorflow as tf
import data
import train
import model
import evaluation
import math


# general settings
EVAL_MEASURES   = ['ClasswiseAccuracy', 'ClasswiseRecall', 'ClasswisePrecision', 'ClasswiseF1']
CLASSNAMES      = ['crack', 'inactive']
NUM_CLASSES     = len(CLASSNAMES)
DATASET_TRAIN   = Path('.') / 'data' / 'train.csv'
#DATASET_TRAIN   = Path("D:\\Winnie\\Documents\\master\\Nuremburg_University\\2018WS\DL\\5\\src_toimplement\\data\\train.csv")
# training related settings
# TODO adapt these to your needs and find suitable hyperparameter settings
MODEL           = 'alexnet'
BATCH_SIZE      = 10
LEARNING_RATE   = (1e-4)
SAVE_DIR        = Path('out')
STOP_PATIENCE   = 15            # wait for the loss to increase for this many epochs until training stops
TRAIN_PART      = 0.8
SHUFFLE         = True
AUGMENT         = True
THRESHOLD       = 0.7


if __name__ == '__main__':

    ds = data.Dataset(DATASET_TRAIN, CLASSNAMES, BATCH_SIZE, SHUFFLE, AUGMENT)
    ds_train, ds_validation = ds.split_train_validation(TRAIN_PART)
    print('Dataset split into {:d} training and {:d} validation samples'.format(len(ds_train), len(ds_validation)))
    prediction_logits = []
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):

        # Note: We need to use placeholders for inputs and outputs. Otherwise,
        # the batch size would be fixed and we could not use the trained model with
        # a different batch size. In addition, the names of these tensors must be "inputs"
        # and "labels" such that we can find them on the evaluation server. DO NOT CHANGE THIS!
        x = tf.placeholder(tf.float32, [None] + [224,224] + [1], 'inputs')
        labels = tf.placeholder(tf.float32, [None] + [NUM_CLASSES], 'labels')
        #is_training = tf.placeholder(tf.bool)
        
        if MODEL == 'alexnet':
            prediction_logits = model.alexnet(x, len(CLASSNAMES),0.5)
        if MODEL == 'resnet':
            prediction_logits = model.resnet(x, len(CLASSNAMES))

        # apply loss

        # TODO : Implement suitable loss function for multi-label problem
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=prediction_logits))
        # convert into binary (boolean) predictions
        prediction_bin = tf.round(tf.nn.sigmoid(prediction_logits))# TODO
        #prediction_bin = tf.nn.sigmoid(prediction_logits)
        #prediction_bin = tf.cast(tf.greater(tf.nn.sigmoid(prediction_logits), THRESHOLD), tf.int32)# TODO


        # Note: The name of the predictions tensor needs to be "predictions" such that
        # we can identify it when loading the model on the evaluation server.
        # We use tf.identity to give it a fixed name.
        prediction = tf.identity(prediction_bin, name='predictions')

    # Note: this is required to update batchnorm during training..
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        ev = evaluation.create_evaluation(EVAL_MEASURES, CLASSNAMES)
        train_batch = math.ceil(ds_train.__len__()/ float(BATCH_SIZE))
        valid_batch = math.ceil(ds_validation.__len__()/ float(BATCH_SIZE))
        trainer = train.Trainer(loss, prediction, optimizer, ds_train, ds_validation,train_batch, valid_batch, STOP_PATIENCE, ev, x, labels)


    # check if output directory does not exist
    if SAVE_DIR.exists():
        print('directory {} must not exist..'.format(str(SAVE_DIR)))
        exit(1)


    with tf.Session() as sess:
        sess.run(tf.initializers.global_variables())

        # train
        print('Run training loop')
        trainer.run(sess)

        # save
        print('Save model')
        sh.simple_save(sess, str(SAVE_DIR), inputs = {'x': x}, outputs = {'y': prediction})

        # create zip file for submission
        print('Create zip file for submission')
        sh.zip_dir(SAVE_DIR, SAVE_DIR / 'model.zip')


