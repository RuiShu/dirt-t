import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
PATH = '/home/ruishu/data'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src',    type=str,   default='mnist',   help="Src data")
parser.add_argument('--trg',    type=str,   default='svhn',    help="Trg data")
parser.add_argument('--nn',     type=str,   default='small',   help="Architecture")
parser.add_argument('--trim',   type=int,   default=5,         help="Trim")
parser.add_argument('--inorm',  type=int,   default=1,         help="Instance normalization flag")
parser.add_argument('--radius', type=float, default=3.5,       help="Perturbation 2-norm ball radius")
parser.add_argument('--dw',     type=float, default=1e-2,      help="Domain weight")
parser.add_argument('--bw',     type=float, default=1e-2,      help="Beta (KL) weight")
parser.add_argument('--sw',     type=float, default=1,         help="Src weight")
parser.add_argument('--tw',     type=float, default=1e-2,      help="Trg weight")
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--dirt',   type=int,   default=0,         help="0 == VADA, >0 == DIRT-T interval")
parser.add_argument('--run',    type=int,   default=999,       help="Run index. >= 999 == debugging")
parser.add_argument('--datadir',type=str,   default=PATH,      help="Data directory")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
codebase_args.args = args = parser.parse_args()

# Argument overrides and additions
src2Y = {'mnist': 10, 'mnistm': 10, 'digit': 10, 'svhn': 10, 'cifar': 9, 'stl': 9, 'sign': 43}
args.Y = src2Y[args.src]
args.H = 32
args.bw = args.bw if args.dirt > 0 else 0.  # mask bw when running VADA
pprint(vars(args))

from codebase.models.dirtt import dirtt
from codebase.train import train
from codebase.datasets import get_data

# Make model name
setup = [
    ('model={:s}',  'dirtt'),
    ('src={:s}',    args.src),
    ('trg={:s}',    args.trg),
    ('nn={:s}',     args.nn),
    ('trim={:d}',   args.trim),
    ('dw={:.0e}',   args.dw),
    ('bw={:.0e}',   args.bw),
    ('sw={:.0e}',   args.sw),
    ('tw={:.0e}',   args.tw),
    ('dirt={:05d}', args.dirt),
    ('run={:04d}',  args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = dirtt()
M.sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if args.dirt > 0:
    run = args.run if args.run < 999 else 0
    setup = [
        ('model={:s}',  'dirtt'),
        ('src={:s}',    args.src),
        ('trg={:s}',    args.trg),
        ('nn={:s}',   args.nn),
        ('trim={:d}',   args.trim),
        ('dw={:.0e}',   args.dw),
        ('bw={:.0e}',  0),
        ('sw={:.0e}',  args.sw),
        ('tw={:.0e}',  args.tw),
        ('dirt={:05d}', 0),
        ('run={:04d}',  run)
    ]
    vada_name = '_'.join([t.format(v) for (t, v) in setup])
    path = tf.train.latest_checkpoint(os.path.join('checkpoints', vada_name))
    saver.restore(M.sess, path)
    print "Restored from {}".format(path)

src = get_data(args.src)
trg = get_data(args.trg)

train(M, src, trg,
      saver=saver,
      has_disc=args.dirt == 0,
      model_name=model_name)
