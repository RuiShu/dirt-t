"""
Dummy module where global parameters are set and stored
Kinda hacky. I'll switch to tf.FLAGS at some point >.>
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
args = None
