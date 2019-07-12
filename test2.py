import os
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
from nmt import Encoder, Decoder, BahdanauAttention, loss_function, define_checkpoints, train_step, train, decode 
import math
import time


print('reading vocabulary file: %s' % args['--vocab'])
VOCAB = Vocab.load('VOCAB_FILE')

print('reading source sentence file: %s' % args['--src-file'])
src_sents = read_file('train.es', source='src')

print('reading target sentence file: %s' % args['--tgt-file'])
tgt_sents = read_file('train.en', source='tgt')

print("padding sequences...")
src_pad = VOCAB.src.to_input_tensor(src_sents)
tgt_pad = VOCAB.tgt.to_input_tensor(tgt_sents)

print("defining parameters...")
EMBED_SIZE = 300
HIDDEN_SIZE = 300
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
NUM_TRAIN_STEPS = 2
BUFFER_SIZE = len(src_pad)
steps_per_epoch = len(src_pad)//BATCH_SIZE
vocab_inp_size = len(VOCAB.src) +1
vocab_tar_size = len(VOCAB.tgt) +1


encoder = Encoder(vocab_inp_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
decoder.attention.W1.layer  = Decoder(vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)    
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                     encoder=encoder,
                     decoder=decoder)

encoder

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


tf.train.list_variables(tf.train.latest_checkpoint('./training_checkpoints'))



