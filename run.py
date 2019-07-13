#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train --vocab=<file> --src-file=<file> --tgt-file=<file>
    run.py batch_decode MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode MODEL_PATH --TEST_SENTENCE=<sent>
"""
import os

os.chdir('C:/Users/u346442/Documents/Stuffs/Deep Learning/Neural Machine Translation/Neural-Machine-Translation')

import os
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
from nmt import Encoder, Decoder, Attention, NMT, define_checkpoints, train_step, train, decode 
import math
import time

if __name__ == '__main__':
    
    args = docopt(__doc__)
    
    print('reading vocabulary file: %s' % args['--vocab'])
    VOCAB = Vocab.load(args['--vocab'])
    
    print('reading source sentence file: %s' % args['--src-file'])
    src_sents = read_file(args['--src-file'], source='src')
    
    print('reading target sentence file: %s' % args['--tgt-file'])
    tgt_sents = read_file(args['--tgt-file'], source='tgt')
    
    print("padding sequences...")
    src_pad = VOCAB.src.to_input_tensor(src_sents)
    tgt_pad = VOCAB.tgt.to_input_tensor(tgt_sents)
    
    print("defining parameters...")
    EMBED_SIZE = 256
    HIDDEN_SIZE = 256
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    NUM_TRAIN_STEPS = 2
    BUFFER_SIZE = len(src_pad)
    steps_per_epoch = len(src_pad)//BATCH_SIZE
    vocab_inp_size = len(VOCAB.src) +1
    vocab_tar_size = len(VOCAB.tgt) +1
    
    print("preparing data pipline...") 
    dataset = tf.data.Dataset.from_tensor_slices((src_pad, tgt_pad)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    print("deleting files not required...") 
    del src_sents,tgt_sents,src_pad, tgt_pad
    
    if args['train']:
        print("beginning training...")
        train(dataset, 
              EMBED_SIZE, 
              HIDDEN_SIZE, 
              DROPOUT_RATE, 
              BATCH_SIZE, 
              NUM_TRAIN_STEPS, 
              BUFFER_SIZE,
              steps_per_epoch,
              vocab_inp_size,
              vocab_tar_size,
              VOCAB)
    
    if args['batch_decode']:        
        raise NotImplementedError
    
    if args['decode']:
        
        print('restoring pre-trained model')
        encoder = Encoder(vocab_inp_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)    
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
        print("beginning decoding...")
        decode(args['--TEST_SENTENCE'])
        
    print("training complete!")