#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    nmt.py --vocab=<file> --src-file=<file> --tgt-file=<file>
"""

import os
from docopt import docopt
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from vocab import *
import math
import time


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def define_checkpoints(optimizer, encoder, decoder):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    return checkpoint, checkpoint_prefix

@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, attention, optimizer , loss_object):
  loss = 0
  
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    
    dec_input = tf.expand_dims([VOCAB.tgt['<s>']] * BATCH_SIZE, 1)
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions, loss_object)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def train():
    
    print("initializing seq2seq model...")
    print("initializing seq2seq model... encoder")
    encoder = Encoder(vocab_inp_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
    print("initializing seq2seq model... decoder")
    decoder = Decoder(vocab_tar_size, EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE)
    print("initializing seq2seq model... attention layer")
    attention_layer = BahdanauAttention(10)
    print("initializing seq2seq model... optimizer")
    optimizer = tf.keras.optimizers.Adam()
    print("initializing seq2seq model... loss")
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    print("initializing seq2seq model... defining checkpoint")
    checkpoint, checkpoint_prefix = define_checkpoints(optimizer, encoder, decoder)
    
    for epoch in range(NUM_TRAIN_STEPS):
      start = time.time()
    
      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0
    
      for (batch, (inp, targ)) in enumerate(dataset.take(BUFFER_SIZE)):
    
        batch_loss = train_step(inp, targ, enc_hidden, encoder = encoder, 
                                decoder = decoder, attention = attention_layer, 
                                optimizer = optimizer, loss_object = loss_object)
        total_loss += batch_loss
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / BATCH_SIZE))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
  
  
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
    EMBED_SIZE = 300
    HIDDEN_SIZE = 1024
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 64
    NUM_TRAIN_STEPS = 100
    BUFFER_SIZE = len(src_pad)
    steps_per_epoch = len(src_pad)//BATCH_SIZE
    vocab_inp_size = len(VOCAB.src) +1
    vocab_tar_size = len(VOCAB.tgt) +1
    
    print("preparing data pipline...") 
    dataset = tf.data.Dataset.from_tensor_slices((src_pad, tgt_pad)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    print("deleting files not required...") 
    del src_sents,tgt_sents,src_pad, tgt_pad
    
    print("beginning training...")
    train()
    
    print("training complete!")