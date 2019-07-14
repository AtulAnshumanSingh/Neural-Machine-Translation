import math
from typing import List
import numpy as np 
import tensorflow as tf
import os    

def read_file(filepath, source):

    data = []
    
    for line in open(filepath):
        
        sentence = line.strip().split(' ')
        """ append "<s> and </s> only to target source """
        
        if source == 'tgt':
            sentence = ['<s>'] + sentence + ['</s>']
        
        data.append(sentence)
    
    return data

def read_sent(sent, source):
    
    data = []
        
    sentence = sent.strip().split(' ')
    """ append "<s> and </s> only to target source """
    
    if source == 'tgt':
        sentence = ['<s>'] + sentence + ['</s>']
    
    data.append(sentence)
    
    return data

def pad_sents(sents, pad_token):

    sents_padded = []
    
    max_len = 0
        
    for sent in sents:
        
        if len(sent) > max_len:
            max_len = len(sent)
    
    sent = None
    
    for sent in sents:
        
        if len(sent) < max_len:
            sent = sent + [pad_token] * (max_len - len(sent))
        
        sents_padded.append(sent) 
    
    return sents_padded

def batch_iter(data, batch_size, shuffle=False):

    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def mask_fill_inf(matrix, mask):
    
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (-((mask * num + num) - num)) + (matrix * negmask)
    
def gerenrate_mask(enc_masks, source_lengths, enc_hiddens):
    
    masks_ = np.zeros([enc_hiddens.shape[0], enc_hiddens.shape[1]], dtype = np.float32)
    for ids, _ in enumerate(source_lengths):
        masks_[ids,_:] = 1
        
    return tf.convert_to_tensor(masks_)

def generate_target_mask(matrix, tgt_lengths, BATCH_SIZE):
    
    masks_ = np.ones([BATCH_SIZE, matrix.shape[1]], dtype = np.float32)
    
    for ids, _ in enumerate(tgt_lengths):
        masks_[ids,_-1:] = 0
        
    return tf.convert_to_tensor(masks_)

def define_checkpoints(optimizer, model):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model = model)
    return checkpoint, checkpoint_prefix
    