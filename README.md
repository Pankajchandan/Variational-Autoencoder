# Variational-Autoencoder
Tensorflow framework for variational convolutional autoencoder


### Get a list of jpg file (Only JPG works!)
files = [os.path.join(some_dir, file_i) for file_i in os.listdir(some_dir) if file_i.endswith('.jpg')]

### Train by changing these parameters!(first covert model.ipynb to .py file)
model.train_vae(files,
              input_shape,
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              ckpt_name="vae.ckpt")
