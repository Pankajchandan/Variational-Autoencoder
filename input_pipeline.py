
# coding: utf-8

# In[2]:


import os
import pickle
import numpy as np
import tensorflow as tf


# In[3]:


def input_pipeline(files, batch_size, n_epochs, shape, crop_shape=None,
                          crop_factor=1.0, n_threads=4):
    """Creates a pipefile from a list of image files.
    Includes batch generator/central crop/resizing options.
    The resulting generator will dequeue the images batch_size at a time until
    it throws tf.errors.OutOfRangeError when there are no more images left in
    the queue.

    Parameters
    ----------
    files : list
        List of paths to image files.
    batch_size : int
        Number of image files to load at a time.
    n_epochs : int
        Number of epochs to run before raising tf.errors.OutOfRangeError
    shape : list
        [height, width, channels]
    crop_shape : list
        [height, width] to crop image to.
    crop_factor : float
        Percentage of image to take starting from center.
    n_threads : int, optional
        Number of threads to use for batch shuffling
    """

    # We first create a "producer" queue.  It creates a production line which
    # will queue up the file names and allow another queue to deque the file
    # names all using a tf queue runner.
    # Put simply, this is the entry point of the computational graph.
    # It will generate the list of file names.
    # We also specify it's capacity beforehand.
    producer = tf.train.string_input_producer(
        files, capacity=len(files))

    # We need something which can open the files and read its contents.
    reader = tf.WholeFileReader()

    # We pass the filenames to this object which can read the file's contents.
    # This will create another queue running which dequeues the previous queue.
    keys, vals = reader.read(producer)

    # And then have to decode its contents as we know it is a jpeg image
    imgs = tf.image.decode_jpeg(
        vals,
        channels=3 if len(shape) > 2 and shape[2] == 3 else 0)

    # explicitly define the shape of the tensor.
    # This is because the decode_jpeg operation is still a node in the graph
    # and doesn't yet know the shape of the image.  Future operations however
    # need explicit knowledge of the image's shape in order to be created.
    imgs.set_shape(shape)

    # centrally crop the image to the size of 100x100.
    # This operation required explicit knowledge of the image's shape.
    if shape[0] > shape[1]:
        rsz_shape = [int(shape[0] / shape[1] * crop_shape[0] / crop_factor),
                     int(crop_shape[1] / crop_factor)]
    else:
        rsz_shape = [int(crop_shape[0] / crop_factor),
                     int(shape[1] / shape[0] * crop_shape[1] / crop_factor)]
    rszs = tf.image.resize_images(imgs, rsz_shape)
    crops = (tf.image.resize_image_with_crop_or_pad(
        rszs, crop_shape[0], crop_shape[1])
        if crop_shape is not None
        else imgs)

    # Now we'll create a batch generator that will also shuffle our examples.
    # We tell it how many it should have in its buffer when it randomly
    # permutes the order.
    min_after_dequeue = len(files) // 10

    # The capacity should be larger than min_after_dequeue, and determines how
    # many examples are prefetched.  TF docs recommend setting this value to:
    # min_after_dequeue + (num_threads + a small safety margin) * batch_size
    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    # Randomize the order and output batches of batch_size.
    batch = tf.train.shuffle_batch([crops],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads)

    # alternatively, we could use shuffle_batch_join to use multiple reader
    # instances, or set shuffle_batch's n_threads to higher than 1.

    return batch


# In[4]:


import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


# In[5]:


def batch_norm(x, phase_train, name='bn', decay=0.99, reuse=None, affine=True):
    """
    Batch normalization on convolutional maps.
    Only modified to infer shape from input tensor x.
    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    name
        string, variable name
    affine
        whether to affine-transform outputs
    Return
    ------
    normed
        batch-normalized maps
    """
    with tf.variable_scope(name, reuse=reuse):
        og_shape = x.get_shape().as_list()
        if len(og_shape) == 2:
            x = tf.reshape(x, [-1, 1, 1, og_shape[1]])
        shape = x.get_shape().as_list()
        beta = tf.get_variable(name='beta', shape=[shape[-1]],
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[shape[-1]],
                                initializer=tf.constant_initializer(1.0),
                                trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary
            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        # tf.nn.batch_normalization
        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, 1e-5, affine)
        if len(og_shape) == 2:
            normed = tf.reshape(normed, [-1, og_shape[-1]])
    return normed


# In[ ]:




