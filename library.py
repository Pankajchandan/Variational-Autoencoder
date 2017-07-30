
# coding: utf-8

# In[2]:


from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
from scipy.io import wavfile
from scipy.misc import imsave


# In[1]:


def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


# In[3]:


def to_tensor(x):
    """Convert 2 dim Tensor to a 4 dim Tensor ready for convolution.

    Performs the opposite of flatten(x).  If the tensor is already 4-D, this
    returns the same as the input, leaving it unchanged.

    Parameters
    ----------
    x : tf.Tesnor
        Input 2-D tensor.  If 4-D already, left unchanged.

    Returns
    -------
    x : tf.Tensor
        4-D representation of the input.

    Raises
    ------
    ValueError
        If the tensor is not 2D or already 4D.
    """
    if len(x.get_shape()) == 2:
        n_input = x.get_shape().as_list()[1]
        x_dim = np.sqrt(n_input)
        if x_dim == int(x_dim):
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1], name='reshape')
        elif np.sqrt(n_input / 3) == int(np.sqrt(n_input / 3)):
            x_dim = int(np.sqrt(n_input / 3))
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 3], name='reshape')
        else:
            x_tensor = tf.reshape(
                x, [-1, 1, 1, n_input], name='reshape')
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    return x_tensor


# In[4]:


def conv2d(x, n_output,
           k_h=5, k_w=5, d_h=2, d_w=2,
           padding='SAME', name='conv2d', reuse=None):
    """Helper for creating a 2d convolution operation.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of convolution
    """
    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, x.get_shape()[-1], n_output],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(
            name='conv',
            input=x,
            filter=W,
            strides=[1, d_h, d_w, 1],
            padding=padding)

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=conv,
            bias=b)

    return h, W


# In[5]:


def deconv2d(x, n_output_h, n_output_w, n_output_ch, n_input_ch=None,
             k_h=5, k_w=5, d_h=2, d_w=2,
             padding='SAME', name='deconv2d', reuse=None):
    """Deconvolution helper.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to convolve.
    n_output_h : int
        Height of output
    n_output_w : int
        Width of output
    n_output_ch : int
        Number of filters.
    k_h : int, optional
        Kernel height
    k_w : int, optional
        Kernel width
    d_h : int, optional
        Height stride
    d_w : int, optional
        Width stride
    padding : str, optional
        Padding type: "SAME" or "VALID"
    name : str, optional
        Variable scope

    Returns
    -------
    op : tf.Tensor
        Output of deconvolution
    """
    with tf.variable_scope(name or 'deconv2d', reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[k_h, k_w, n_output_ch, n_input_ch or x.get_shape()[-1]],
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d_transpose(
            name='conv_t',
            value=x,
            filter=W,
            output_shape=tf.stack(
                [tf.shape(x)[0], n_output_h, n_output_w, n_output_ch]),
            strides=[1, d_h, d_w, 1],
            padding=padding)

        conv.set_shape([None, n_output_h, n_output_w, n_output_ch])

        b = tf.get_variable(
            name='b',
            shape=[n_output_ch],
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h', value=conv, bias=b)

    return h, W


# In[6]:


def lrelu(features, leak=0.2):
    """Leaky rectifier.

    Parameters
    ----------
    features : tf.Tensor
        Input to apply leaky rectifier to.
    leak : float, optional
        Percentage of leak.

    Returns
    -------
    op : tf.Tensor
        Resulting output of applying leaky rectifier activation.
    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * features + f2 * abs(features)


# In[7]:


def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    h, W : tf.Tensor, tf.Tensor
        Output of fully connected layer and the weight matrix
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name='b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


# In[9]:


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


# In[10]:


def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    imsave(arr=np.squeeze(m), name=saveto)
    return m


# In[11]:


def make_latent_manifold(corners, n_samples):
    """Create a 2d manifold out of the provided corners: n_samples * n_samples.

    Parameters
    ----------
    corners : list of np.ndarray
        The four corners to intepolate.
    n_samples : int
        Number of samples to use in interpolation.

    Returns
    -------
    arr : np.ndarray
        Stacked array of all 2D interpolated samples
    """
    left = interp(corners[0], corners[1], n_samples)
    right = interp(corners[2], corners[3], n_samples)

    embedding = []
    for row_i in range(n_samples):
        embedding.append(interp(left[row_i], right[row_i], n_samples))
    return np.vstack(embedding)


# In[ ]:




