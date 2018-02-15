from args import args
import tensorflow as tf
import shutil
import os
import datasets
import numpy as np

def u2t(x):
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    return x * 2 - 1

def delete_existing(path):
    if args.run < 999:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def save_accuracy(M, fn_acc_key, tag, dataloader,
                  train_writer=None, step=None, print_list=None,
                  full=True):
    fn_acc = getattr(M, fn_acc_key, None)
    if fn_acc:
        acc, summary = exact_accuracy(fn_acc, tag, dataloader, full)
        train_writer.add_summary(summary, step + 1)
        print_list += [os.path.basename(tag), acc]

def exact_accuracy(fn_acc, tag, dataloader, full=True):
    # Fixed shuffling scheme
    state = np.random.get_state()
    np.random.seed(0)
    shuffle = np.random.permutation(len(dataloader.images))
    np.random.set_state(state)

    xs = dataloader.images[shuffle]
    ys = dataloader.labels[shuffle] if dataloader.labels is not None else None

    if not full:
        xs = xs[:1000]
        ys = ys[:1000] if ys is not None else None

    acc = 0.
    n = len(xs)
    bs = 200

    for i in xrange(0, n, bs):
        x = u2t(xs[i:i+bs]) if dataloader.cast else xs[i:i+bs]
        y = ys[i:i+bs] if ys is not None else dataloader.labeler(x)
        acc += fn_acc(x, y) * len(x) / n

    summary = tf.Summary.Value(tag=tag, simple_value=acc)
    summary = tf.Summary(value=[summary])
    return acc, summary
