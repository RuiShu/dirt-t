import os
import sys
import argparse
from codebase import args as codebase_args
from pprint import pprint
import tensorflow as tf

# Settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--src',    type=str,   default='mnist32', help="Src data")
parser.add_argument('--trg',    type=str,   default='svhn',    help="Trg data")
parser.add_argument('--design', type=str,   default='v11_y',   help="Architecture design")
parser.add_argument('--trim',   type=int,   default=5,         help="Trim")
parser.add_argument('--pert',   type=str,   default='vat',     help="Type of perturbation")
parser.add_argument('--ball',   type=float, default=3.5,       help="Perturbation 2-norm ball radius")
parser.add_argument('--dw',     type=float, default=1e-2,      help="Domain weight")
parser.add_argument('--sw',     type=float, default=1,         help="Src weight")
parser.add_argument('--tw',     type=float, default=1e-2,      help="Trg weight")
parser.add_argument('--bw',     type=float, default=1e-2,      help="Beta (KL) weight")
parser.add_argument('--lr',     type=float, default=1e-3,      help="Learning rate")
parser.add_argument('--dirt',   type=int,   default=0,         help="0 == VADA, >0 == DIRT-T interval")
parser.add_argument('--run',    type=int,   default=999,       help="Run index")
parser.add_argument('--logdir', type=str,   default='log',     help="Log directory")
codebase_args.args = args = parser.parse_args()
pprint(vars(args))

from codebase.models.dirtt import dirtt
from codebase.train import train
from codebase.utils import get_data

# Make model name
setup = [
    ('model={:s}',  'dirtt'),
    ('src={:s}',  args.src),
    ('trg={:s}',  args.trg),
    ('des={:s}',  args.design),
    ('trim={:d}', args.trim),
    ('dw={:.0e}',  args.dw),
    ('cw={:.0e}',  args.cw),
    ('sbw={:.0e}', args.sbw),
    ('tbw={:.0e}', args.tbw),
    ('dirt={:05d}', args.dirt),
    ('run={:04d}',   args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in setup])
print "Model name:", model_name

M = dirtt()
M.sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if args.dirt > 0:
    # Figure out later
    pass

src = get_data(args.src)
trg = get_data(args.trg)

train(M, src, trg,
      saver=saver,
      has_disc=True,
      add_z=True,
      model_name=model_name)
