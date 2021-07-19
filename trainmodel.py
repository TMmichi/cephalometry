import tensorflow as tf
import numpy as np
from dataset_function import *
from landmark_data import *

####################################################################
#CONSTANTS
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.001
DROPOUT_SWITCH = True
gpu_num  ="0"
#SWITCH
seperate = True
####################################################################

tf.logging.set_verbosity(tf.logging.INFO)

def modeling(features, labels, mode):
    if seperate:
        for d in ["/gpu:"+gpu_num]:
            with tf.device(d):
                random.seed(datetime.datetime.now())
                seed = random.randint(0,12345)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    status = True
                if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
                    status = False

                input_layer = tf.reshape(features["x"],[-1,IMAGE_SIZE,IMAGE_SIZE,1])

                batch1 = tf.layers.batch_normalization(inputs = input_layer,
                                                       axis = -1,
                                                       scale = True,
                                                       training = status)
#                print("batch1 = ", batch1.shape)

                conv1 = tf.layers.conv2d(inputs = batch1,
                                         filters = 32,
                                         kernel_size = 10,
                                         strides = (1,1),
                                         padding = 'valid',
                                         activation = tf.nn.relu)
#                print("conv1 = ", conv1.shape)

                pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                                pool_size = [2,2],
                                                strides = 2)
#               print("pool1 = ", pool1.shape)

                dropout1 = tf.layers.dropout(inputs = pool1, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

                batch2 = tf.layers.batch_normalization(inputs = dropout1,
                                                       axis = -1,
                                                       scale = True,
                                                       training = status)
#               print("batch2 = ", batch2.shape)

                conv2 = tf.layers.conv2d(inputs = batch2,
                                         filters = 64,
                                         kernel_size = 7,
                                         strides = (1,1),
                                         padding = 'valid',
                                         activation = tf.nn.relu)
#               print("conv2 = ", conv2.shape)

                pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                                pool_size = [2,2],
                                                strides = 2)
#               print("pool2 = ", pool2.shape)

                dropout2 = tf.layers.dropout(inputs = pool2, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

                batch3 = tf.layers.batch_normalization(inputs = dropout2,
                                                       axis = -1,
                                                       scale = True,
                                                       training = status)
#               print("batch3 = ", batch3.shape)

                conv3 = tf.layers.conv2d(inputs = batch3,
                                         filters = 128,
                                         kernel_size = 5,
                                         strides = (2,2),
                                         padding = 'valid',
                                         activation = tf.nn.relu)
#               print("conv3 = ", conv3.shape)

                pool3 = tf.layers.max_pooling2d(inputs = conv3,
                                                pool_size = [2,2],
                                                strides = 2)
#               print("pool3 = ", pool3.shape)

                dropout3 = tf.layers.dropout(inputs = pool3, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

                batch4 = tf.layers.batch_normalization(inputs = dropout3,
                                                       axis = -1,
                                                       scale = True,
                                                       training = status)
#               print("batch4 = ", batch4.shape)

                batch_size, height, width, depth = batch4.shape[:5]

                conv4 = tf.layers.conv2d(inputs = batch4,
                                         filters = 256,
                                         kernel_size = [height,width],
                                         strides = (1,1),
                                         padding = 'valid',
                                         activation = tf.nn.relu)
#               print("conv4 = ", conv4.shape)

                dropout4 = tf.layers.dropout(inputs = conv4, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

                conv4_flat = tf.reshape(dropout4,[-1,1*1*256])
#               print("conv4_flat = ", conv4_flat.shape)

                dense = tf.layers.dense(inputs = conv4_flat,
                                        units = 256,
                                        activation = tf.nn.relu)

                #dropout
                dropout = tf.layers.dropout(inputs = dense, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

                logits = tf.layers.dense(inputs = dropout, units = 2)

                #export the class which has its value to be maximum (true landmark or false)
                predictions = {
                    "classes": tf.argmax(input = logits, axis = 1),
                    "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
                    }

                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)

                onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 2)

                #loss = softmax_cross_entropy, logits -> dimension: 2
                loss = tf.losses.softmax_cross_entropy(
                    onehot_labels = onehot_labels, logits = logits)
                + WEIGHT_DECAY * tf.nn.l2_loss(conv1)
                + WEIGHT_DECAY * tf.nn.l2_loss(conv2)
                + WEIGHT_DECAY * tf.nn.l2_loss(conv3)
                + WEIGHT_DECAY * tf.nn.l2_loss(conv4)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    print("Training...")
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    logging_hook = tf.train.LoggingTensorHook({"loss":loss},every_n_iter=10)
                    with tf.control_dependencies(update_ops):
                        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op, training_hooks=[logging_hook])

                eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                        labels=labels, predictions=predictions["classes"])}

                print("Evaluating...")
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    else:
        random.seed(datetime.datetime.now())
        seed = random.randint(0,12345)
        if mode == tf.estimator.ModeKeys.TRAIN:
            status = True
        if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
            status = False

        input_layer = tf.reshape(features["x"],[-1,IMAGE_SIZE,IMAGE_SIZE,1])

        batch1 = tf.layers.batch_normalization(inputs = input_layer,
                                               axis = -1,
                                               scale = True,
                                               training = status)
        print("batch1 = ", batch1.shape)

        conv1 = tf.layers.conv2d(inputs = batch1,
                                 filters = 32,
                                 kernel_size = 10,
                                 strides = (1,1),
                                 padding = 'valid',
                                 activation = tf.nn.relu)
        print("conv1 = ", conv1.shape)

        pool1 = tf.layers.max_pooling2d(inputs = conv1,
                                        pool_size = [2,2],
                                        strides = 2)
        print("pool1 = ", pool1.shape)

        dropout1 = tf.layers.dropout(inputs = pool1, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

        batch2 = tf.layers.batch_normalization(inputs = dropout1,
                                               axis = -1,
                                               scale = True,
                                               training = status)
        print("batch2 = ", batch2.shape)

        conv2 = tf.layers.conv2d(inputs = batch2,
                                 filters = 64,
                                 kernel_size = 7,
                                 strides = (1,1),
                                 padding = 'valid',
                                 activation = tf.nn.relu)
        print("conv2 = ", conv2.shape)

        pool2 = tf.layers.max_pooling2d(inputs = conv2,
                                        pool_size = [2,2],
                                        strides = 2)
        print("pool2 = ", pool2.shape)

        dropout2 = tf.layers.dropout(inputs = pool2, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

        batch3 = tf.layers.batch_normalization(inputs = dropout2,
                                               axis = -1,
                                               scale = True,
                                               training = status)
        print("batch3 = ", batch3.shape)

        conv3 = tf.layers.conv2d(inputs = batch3,
                                 filters = 128,
                                 kernel_size = 5,
                                 strides = (2,2),
                                 padding = 'valid',
                                 activation = tf.nn.relu)
        print("conv3 = ", conv3.shape)

        pool3 = tf.layers.max_pooling2d(inputs = conv3,
                                        pool_size = [2,2],
                                        strides = 2)
        print("pool3 = ", pool3.shape)

        dropout3 = tf.layers.dropout(inputs = pool3, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

        batch4 = tf.layers.batch_normalization(inputs = dropout3,
                                               axis = -1,
                                               scale = True,
                                               training = status)
        print("batch4 = ", batch4.shape)

        batch_size, height, width, depth = batch4.shape[:5]

        conv4 = tf.layers.conv2d(inputs = batch4,
                                 filters = 256,
                                 kernel_size = [height,width],
                                 strides = (1,1),
                                 padding = 'valid',
                                 activation = tf.nn.relu)
        print("conv4 = ", conv4.shape)

        dropout4 = tf.layers.dropout(inputs = conv4, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

        conv4_flat = tf.reshape(dropout4,[-1,1*1*256])
        print("conv4_flat = ", conv4_flat.shape)

        dense = tf.layers.dense(inputs = conv4_flat,
                                units = 256,
                                activation = tf.nn.relu)

        #dropout
        dropout = tf.layers.dropout(inputs = dense, rate = DROPOUT_RATE, seed = seed, training = DROPOUT_SWITCH)

        logits = tf.layers.dense(inputs = dropout, units = 2)

        #export the class which has its value to be maximum (true landmark or false)
        predictions = {
            "classes": tf.argmax(input = logits, axis = 1),
            "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
            }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)

        onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 2)

        #loss = softmax_cross_entropy, logits -> dimension: 2
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels = onehot_labels, logits = logits)
        + WEIGHT_DECAY * tf.nn.l2_loss(conv1)
        + WEIGHT_DECAY * tf.nn.l2_loss(conv2)
        + WEIGHT_DECAY * tf.nn.l2_loss(conv3)
        + WEIGHT_DECAY * tf.nn.l2_loss(conv4)

        if mode == tf.estimator.ModeKeys.TRAIN:
            print("Training...")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}

        print("Evaluating...")
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
