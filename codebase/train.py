import tensorflow as tf
import tensorbayes as tb
from codebase.args import args
from codebase.datasets import PseudoData
from utils import delete_existing, save_accuracy as save_acc
import os
import sys
import numpy as np

def update_dict(M, feed_dict, src=None, trg=None, bs=100):
    """Update feed_dict with new mini-batch

    M         - (TensorDict) the model
    feed_dict - (dict) tensorflow feed dict
    src       - (obj) source domain. Contains train/test Data obj
    trg       - (obj) target domain. Contains train/test Data obj
    bs        - (int) batch size
    """
    if src:
        src_x, src_y = src.train.next_batch(bs)
        feed_dict.update({M.src_x: src_x, M.src_y: src_y})

    if trg:
        trg_x, trg_y = trg.train.next_batch(bs)
        feed_dict.update({M.trg_x: trg_x, M.trg_y: trg_y})

def train(M, src=None, trg=None, has_disc=True, saver=None, model_name=None):
    """Main training function

    Creates log file, manages datasets, trains model

    M          - (TensorDict) the model
    src        - (obj) source domain. Contains train/test Data obj
    trg        - (obj) target domain. Contains train/test Data obj
    has_disc   - (bool) whether model requires a discriminator update
    saver      - (Saver) saves models during training
    model_name - (str) name of the model being run with relevant parms info
    """
    # Training settings
    bs = 64
    iterep = 1000
    n_epoch = 80
    epoch = 0
    feed_dict = {}

    # Create a log directory and FileWriter
    log_dir = os.path.join(args.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create a save directory
    if saver:
        model_dir = os.path.join('checkpoints', model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    # Replace src domain with psuedolabeled trg
    if args.dirt > 0:
        print "Setting backup and updating backup model"
        src = PseudoData(args.trg, trg, M)
        M.sess.run(M.update_teacher)

        print_list = []
        if src:
            save_acc(M, 'fn_ema_acc', 'test/src_test_ema_1k',
                     src.test,  train_writer, 0, print_list, full=False)

        if trg:
            save_acc(M, 'fn_ema_acc', 'test/trg_test_ema',
                     trg.test,  train_writer, 0, print_list)
            save_acc(M, 'fn_ema_acc', 'test/trg_train_ema_1k',
                     trg.train, train_writer, 0, print_list, full=False)

        print print_list

    if src: print "Src size:", src.train.images.shape
    if trg: print "Trg size:", trg.train.images.shape
    print "Batch size:", bs
    print "Iterep:", iterep
    print "Total iterations:", n_epoch * iterep
    print "Log directory:", log_dir

    for i in xrange(n_epoch * iterep):
        # Run discriminator optimizer
        if has_disc:
            update_dict(M, feed_dict, src, trg, bs)
            summary, _ = M.sess.run(M.ops_disc, feed_dict)
            train_writer.add_summary(summary, i + 1)

        # Run main optimizer
        update_dict(M, feed_dict, src, trg, bs)
        summary, _ = M.sess.run(M.ops_main, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep,
                                            message='{}/{}'.format(epoch, i),
                                            display=args.run >= 999)

        # Log end-of-epoch values
        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            if src:
                save_acc(M, 'fn_ema_acc', 'test/src_test_ema_1k',
                         src.test,  train_writer, i + 1, print_list, full=False)

            if trg:
                save_acc(M, 'fn_ema_acc', 'test/trg_test_ema',
                         trg.test,  train_writer, i + 1, print_list)
                save_acc(M, 'fn_ema_acc', 'test/trg_train_ema_1k',
                         trg.train, train_writer, i + 1, print_list, full=False)

            print_list += ['epoch', epoch]
            print print_list

        # Update pseudolabeler
        if args.dirt and (i + 1) % args.dirt == 0:
            print "Updating teacher model"
            M.sess.run(M.update_teacher)

        if saver and (i + 1) % 20000 == 0:
            save_model(saver, M, model_dir, i + 1)

    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)
