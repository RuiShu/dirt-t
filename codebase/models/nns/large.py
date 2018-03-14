import tensorflow as tf
from codebase.args import args
from codebase.models.extra_layers import leaky_relu, noise
from tensorbayes.layers import dense, conv2d, avg_pool, max_pool, batch_norm, instance_norm
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.ops.nn_ops import dropout

def classifier(x, phase, enc_phase=1, trim=0, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            preprocess = instance_norm if args.inorm else tf.identity
            layout = [
                (preprocess, (), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (noise, (1,), dict(phase=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (avg_pool, (), dict(global_pool=True)),
                (dense, (args.Y,), dict(activation=None))
            ]

            if enc_phase:
                start = 0
                end = len(layout) - trim
            else:
                start = len(layout) - trim
                end = len(layout)

            for i in xrange(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)

    return x

def feature_discriminator(x, phase, C=1, reuse=None):
    with tf.variable_scope('disc/feat', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu): # Switch to leaky?

            x = dense(x, 100)
            x = dense(x, C, activation=None)

    return x
