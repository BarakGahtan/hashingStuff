from torch import nn

import matconvnet2tf_torch
from matconvnet2tf import MatConvNet2TF
import numpy as np

def net(batch_size, hash_size, expected_triplet_count=100, margin=0, weight_decay_factor=0, loss_func=None):
    # Define the neural network model
    class Net(nn.Module):
        def __init__(self, hash_size):
            super(Net, self).__init__()
            # Example: Simple fully connected layer
            # Adjust the architecture as needed
            self.fc = nn.Linear(9216, hash_size)
            # Optional: Initialize weights or add more layers

        def forward(self, t_latent):
            # t_latent: Tensor of shape [batch_size, 9216]
            output = self.fc(t_latent)
            return output

    # Instantiate the model
    model = matconvnet2tf_torch.MatConvNet2PyTorch("data/imagenet-vgg-f_old.mat", input=t_images, ignore=['fc8', 'prob'], do_debug_print=True, input_latent=t_latent, latent_layer="fc6")
    if loss_func is None:

        loss_func = nn.TripletMarginLoss(margin=margin, p=2) # Using Triplet Margin Loss as an example

    model.t_images = t_images
    model.t_latent = t_latent
    model.t_labels = t_labels
    model.t_boolmask = t_boolmask
    model.t_indices_q = t_indices_q
    model.t_indices_p = t_indices_p
    model.t_indices_n = t_indices_n

    fcw = tf.get_variable(name='fc8_custom/weights', shape=[4096, hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)

    fcb = tf.get_variable(name='fc8_custom/biases', shape=[hash_size],
                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                          dtype=tf.float32)

    model.weight_decay_losses.append(tf.abs(tf.reduce_mean(tf.reduce_sum(tf.square(fcw), 0)) - 1.0))
    weight_decay = tf.add_n(model.weight_decay_losses)
    model.weight_decay = weight_decay * weight_decay_factor

    fc8 = tf.nn.bias_add(tf.matmul(model.net['relu7'], fcw), fcb)
    model.output = model.net['fc8_custom'] = fc8
    model.output_norm = tf.nn.l2_normalize(model.output, 1)

    fc8 = tf.nn.bias_add(tf.matmul(model.output2, fcw), fcb)
    model.output_2 = model.net['fc8_custom_2'] = fc8

    model.embedding_var = tf.Variable(tf.zeros((batch_size, hash_size), dtype=tf.float32),
                                      trainable=False,
                                      name='embedding',
                                      dtype='float32')
    model.assignment = tf.assign(model.embedding_var, model.output_norm)

    if loss_func is not None:
        model.loss, model.E = loss_func(model.output, t_indices_q, t_indices_p, t_indices_n, hash_size, batch_size, margin)
        model.loss_2, model.E = loss_func(model.output_2, t_indices_q, t_indices_p, t_indices_n, hash_size, batch_size, margin)

    return model
