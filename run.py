#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train --vocab=<file> --src-file=<file> --tgt-file=<file>
    run.py batch_decode MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode MODEL_PATH --TEST_SENTENCE=<sent>
"""
import os
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
from nmt import Encoder, Decoder, Attention, NMT, define_checkpoints, train_step, train, decode, batch_decode
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
    HIDDEN_SIZE = 512
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 256
    NUM_TRAIN_STEPS = 20
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
        model = NMT(vocab_inp_size, vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
        sample_hidden = model.encoder.initialize_hidden_state()
        sample_output, sample_hidden = model.encoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden)
        sample_decoder_output, _, _ = model.decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                              sample_hidden, sample_output)
        model.load_weights('nmt_model')
        print("beginning decoding...")
        decode(model, args['--TEST_SENTENCE'])
        
    print("training complete!")