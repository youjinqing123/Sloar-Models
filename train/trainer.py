import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self, loss, predictions, optimizer, ds_train, ds_validation, train_batch,valid_batch, stop_patience, evaluation, inputs, labels):
        '''
            Initialize the trainer

            Args:
                loss        	an operation that computes the loss
                predictions     an operation that computes the predictions for the current
                optimizer       optimizer to use
                ds_train        instance of Dataset that holds the training data
                ds_validation   instance of Dataset that holds the validation data
                stop_patience   the training stops if the validation loss does not decrease for this number of epochs
                evaluation      instance of Evaluation
                inputs          placeholder for model inputs
                labels          placeholder for model labels
        '''

        self._train_op = optimizer.minimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._ds_train = ds_train
        self._ds_validation = ds_validation
        self._iter_train = self._ds_train.__iter__()
        self._iter_validation = self._ds_validation.__iter__()
        self._stop_patience = stop_patience
        self._evaluation = evaluation
        self._validation_losses = []
        self._train_losses=[]
        self._model_inputs = inputs
        self._model_labels = labels
        self._train_batch = train_batch
        self._valid_batch = valid_batch
        self._train_mean=[]
        self._valid_mean=[]
        self._greater_num=0
        #self._model_is_training = is_training



        with tf.variable_scope('model', reuse = True):
            self._model_is_training = tf.get_variable('is_training', dtype = tf.bool)

    def _train_epoch(self, sess):
        '''
            trains for one epoch and prints the mean training loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''
        
        # TODO
        #print("train:")
        
        self._iter_train = iter(self._ds_train)
        self._train_losses=[]

        for num in range(self._train_batch):

            batch_x, batch_label = next(self._iter_train)
            

            sess.run(self._train_op,
                     feed_dict={self._model_inputs: batch_x, self._model_labels: batch_label, self._model_is_training: True})
            loss = sess.run(self._loss,
                    feed_dict={self._model_inputs: batch_x, self._model_labels: batch_label, self._model_is_training: True})
            self._train_losses.append(loss)
            #print("batch_label:",batch_label)
            #print("prediction:",prediction)
        print('loss_train:', sum(self._train_losses)/float(len(self._train_losses)))
        self._train_mean.append(sum(self._train_losses)/float(len(self._train_losses)))
         
        



    def _valid_step(self, sess):
        '''
            run the validation and print evalution + mean validation loss to the commandline

            args:
                sess    the tensorflow session that should be used
        '''

        # TODO
        #print("valid:")
        self._model_is_training
        
        
        self._iter_validation = iter(self._ds_validation)
        self._validation_losses = []
        
        for num in range(self._valid_batch):
            batch_x, batch_label = next(self._iter_validation)
            
            loss = sess.run(self._loss,feed_dict={self._model_inputs: batch_x, self._model_labels: batch_label, self._model_is_training: False})
            prediction = sess.run(self._predictions,
                                        feed_dict={self._model_inputs: batch_x, self._model_labels: batch_label, self._model_is_training: False})
            self._validation_losses.append(loss)
            self._evaluation.add_batch(prediction, batch_label)
            
            #print("batch_label:",batch_label)
            #print("prediction:",prediction)
            
        print('loss_valid:', sum(self._validation_losses)/float(len(self._validation_losses)))
        self._valid_mean.append(sum(self._validation_losses)/float(len(self._validation_losses)))
        self._evaluation.flush()
        #print("888888888888888888888888888888888888888888")
        

    def _should_stop(self):
        '''
            determine if training should stop according to stop_patience
        '''
        
        if self._valid_mean[-1] > min(self._valid_mean):
            self._greater_num += 1 
            
        if self._greater_num > self._stop_patience:
            
            return True
        else:
            return False
        # TODO




    def run(self, sess, num_epochs = -1):
        '''
            run the training until num_epochs exceeds or the validation loss did not decrease
            for stop_patience epochs

            args:
                sess        the tensorflow session that should be used
                num_epochs  limit to the number of epochs, -1 means not limit
        '''

        # initial validation step
        self._valid_step(sess)

        i = 0

        # training loop
        #self._iter_train = self._ds_train.__iter__()
        #self._iter_validation = self._ds_validation.__iter__()
        self._greater_num=0
        while i < num_epochs or num_epochs == -1:
            print("iter_num:",i)

            self._train_epoch(sess)
            self._valid_step(sess)
            i += 1
            
            if i==12:
                break

            if self._should_stop():
                break
            
        #plot
        x = range(len(self._train_mean))
        plt.plot(x,self._train_mean,'r--')
        plt.plot(x,self._valid_mean[1:],'b--')
        plt.show()
        
        
    

