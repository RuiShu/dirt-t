import os
import numpy as np
import shutil
import tensorbayes as tb
import tensorflow as tf
from args import args

def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    """Convert [0, 1] float to [-1, 1] float
    """
    return x * 2 - 1

def delete_existing(path):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories
    """
    if args.run < 999:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def save_model(saver, M, model_dir, global_step):
    path = saver.save(M.sess, os.path.join(model_dir, 'model'),
                      global_step=global_step)
    print "Saving model to {}".format(path)

def save_value(fn_val, tag, data,
               train_writer=None, global_step=None, print_list=None,
               full=True):
    """Log fn_val evaluation to tf.summary.FileWriter

    fn_val       - (fn) Takes (x, y) as input and returns value
    tag          - (str) summary tag for FileWriter
    data         - (Data) data object with images/labels attributes
    train_writer - (FileWriter)
    global_step  - (int) global step in file writer
    print_list   - (list) list of vals to print to stdout
    full         - (bool) use full dataset v. first 1000 samples
    """
    acc, summary = compute_value(fn_val, tag, data, full)
    train_writer.add_summary(summary, global_step)
    print_list += [os.path.basename(tag), acc]

def compute_value(fn_val, tag, data, full=True):
    """Compute value w.r.t. data

    fn_val - (fn) Takes (x, y) as input and returns value
    tag    - (str) summary tag for FileWriter
    data   - (Data) data object with images/labels attributes
    full   - (bool) use full dataset v. first 1000 samples
    """
    with tb.nputils.FixedSeed(0):
        shuffle = np.random.permutation(len(data.images))

    xs = data.images[shuffle]
    ys = data.labels[shuffle] if data.labels is not None else None

    if not full:
        xs = xs[:1000]
        ys = ys[:1000] if ys is not None else None

    acc = 0.
    n = len(xs)
    bs = 200

    for i in xrange(0, n, bs):
        x = data.preprocess(xs[i:i+bs])
        y = ys[i:i+bs] if ys is not None else data.labeler(x)
        acc += fn_val(x, y) / n * len(x)

    summary = tf.Summary.Value(tag=tag, simple_value=acc)
    summary = tf.Summary(value=[summary])
    return acc, summary
