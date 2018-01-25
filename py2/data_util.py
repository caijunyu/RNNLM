# -*- coding: utf-8 -*-
import re
import os
import sys
import tensorflow as tf
from best_buckets import calculate_buckets, split_buckets

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def blank_tokenizer(sentence):
    return sentence.strip().split()

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size, tokenizer=None, normalize_digits=False):
    """
    Create vocabulary file (if it does not exist yet) from data file.
    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.
    :param vocabulary_path: path where the vocabulary will be created.
    :param data_paths: data file that will be used to create vocabulary.
    :param max_vocabulary_size: limit on the size of the created vocabulary.
    :param tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
    :param normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """

    if not tf.gfile.Exists(vocabulary_path):
        print('create vocabulary %s from data %s' % (vocabulary_path, ','.join(data_paths)))
        vocab = {}
        for data_path in data_paths:
            with tf.gfile.GFile(data_path, mode = 'rb') as f:
                print('data_path %s' % (data_path))
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 10000 == 0:
                        print('process line %d' % (counter))
                    line = tf.compat.as_bytes(line)
                    tokens = tokenizer(line) if tokenizer else blank_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub(b'0', w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                print('vocab length %d' %(len(vocab)))
        vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with tf.gfile.GFile(vocabulary_path, mode = 'wb') as f:
            for w in vocab_list:
                f.write(w + b'\n')

def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.
    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].
    Args: vocabulary_path: path to the file containing the vocabulary.
    Returns:a pair: the vocabulary (a dictionary mapping string to integers), and
        the reversed vocabulary (a list, which reverses the vocabulary mapping).
    Raises:ValueError: if the provided vocabulary_path does not exist.
    """
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode = 'rb') as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False, with_start=True, with_end=True):
    """
    Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Args:
        sentence: the sentence in bytes format to convert to token-ids.
        vocabulary: a dictionary mapping tokens to integers.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    Returns:a list of integers, the token-ids for the sentence.
    """
    tokens = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence)
    if not normalize_digits:
        ids = [vocabulary.get(w, UNK_ID) for w in tokens]
    else:
        ids = [vocabulary.get(_DIGIT_RE.sub(b'0',w), UNK_ID) for w in tokens]
    if with_start:
        ids = [GO_ID] + ids
    if with_end:
        ids = ids + [EOS_ID]
    return ids


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=False, with_go = True, with_end = True):
    """
    Tokenize data file and turn into token-ids using given vocabulary file.
    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.
    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(target_path):
        print('Tokenizing data in %s' % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, 'rb') as data_file:
            with tf.gfile.GFile(target_path, mode = 'w') as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 10000 == 0:
                        print('tokenizing line %d' % counter)
                    ids = sentence_to_token_ids(tf.compat.as_bytes(line),vocab,tokenizer,normalize_digits)
                    tokens_file.write(' '.join([str(id) for id in ids]) + '\n')



def prepare_data(cache_dir, train_path, dev_path, vocabulary_size):
    vocab_path = os.path.join(cache_dir, 'vocab')
    create_vocabulary(vocab_path, [train_path, dev_path], vocabulary_size)

    train_ids_path = os.path.join(cache_dir, 'train.ids')
    data_to_token_ids(train_path,train_ids_path,vocab_path)

    valid_ids_path = os.path.join(cache_dir, 'valid.ids')
    data_to_token_ids(dev_path, valid_ids_path, vocab_path)
    return train_ids_path, valid_ids_path, vocab_path

def get_vocab_path(cache_dir):
    vocab_path = os.path.join(cache_dir, "vocab")
    return vocab_path

def get_real_vocab_size(vocab_path):
    n = 0
    f = open(vocab_path)
    for line in f:
        n+=1
    f.close()
    return n

#读取目标文件的每一行，将其对应的ids转换成int型存入　data_set中，并记录下这一行句子的长度，存入dat_length_set中。
def read_raw_data(target_path, max_size=None):
    '''
    Args:
        target_path : the path which contains word ids
    '''
    print("read raw data from {}".format(target_path))
    data_set = []
    data_length = []

    with tf.gfile.GFile(target_path, mode="r") as target_file:
        target = target_file.readline()
        counter = 0
        while target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in target.split()]
            data_set.append(target_ids)
            data_length.append(len(target_ids))
            target = target_file.readline()
    return data_set, data_length

#根据bucket切分train和dev数据。
def read_train_dev(cache_dir, train_path, dev_path, vocab_size, max_length, n_bucket):
    train_ids_path, dev_ids_path, vocab_path = prepare_data(cache_dir, train_path, dev_path, vocab_size) #读取train和dev数据,将其建立词典并原文转换成ids
    train_data_set, train_data_length_set = read_raw_data(train_ids_path) #读取ids的训练数据到list中,一个用来存每句话，另一个用来存储每句话的长度。
    dev_data_set, dev_data_length_set = read_raw_data(dev_ids_path)
    all_length_array = train_data_length_set + dev_data_length_set  #将train和dev的所有句子的长度加在一起，用于后续的分buckets
    _buckets = calculate_buckets(all_length_array,max_length,n_bucket)# 根据句子长度list生成bucket
    train_data_bucket,_ = split_buckets(train_data_set,_buckets) # 根据bucket切分　train_data_set
    dev_data_bucket,_ = split_buckets(dev_data_set,_buckets)
    return train_data_bucket, dev_data_bucket, _buckets, vocab_path

#因为之前test的数据没有做过处理，所以在测试的时候也需要将源文件转成ids文件，并读入切分。
def read_test(cache_dir, test_path, vocab_path, max_length, n_bucket):
    global _buckets
    test_ids_path = os.path.join(cache_dir, "test.ids")
    data_to_token_ids(test_path, test_ids_path, vocab_path)
    test_data, test_length = read_raw_data(test_ids_path)
    _buckets = calculate_buckets(test_length, max_length, n_bucket)
    test_data_bucket, test_data_order = split_buckets(test_data, _buckets)
    return test_data_bucket, _buckets, test_data_order

if __name__ == "__main__":
    cache_dir = "/home/robocai/RNNLMCAI/model/model_ptb/data_cache/"
    vocabulary_path = "/home/robocai/RNNLMCAI/model/model_ptb/data_cache/vocab"
    data_paths = ['/home/robocai/RNNLMCAI/data/ptb/train', '/home/robocai/RNNLMCAI/data/ptb/valid']
    train_data_path = '/home/robocai/RNNLMCAI/data/ptb/train'
    valid_data_path = '/home/robocai/RNNLMCAI/data/ptb/valid'
    train_ids_path = '/home/robocai/RNNLMCAI/model/model_ptb/data_cache/train_ids'
    valid_ids_path = '/home/robocai/RNNLMCAI/model/model_ptb/data_cache/valid_ids'
    max_vocabulary_size = 10000
    # create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size, tokenizer=None, normalize_digits=False)
    # data_to_token_ids(train_data_path, train_ids_path, vocabulary_path)
    # data_to_token_ids(valid_data_path, valid_ids_path, vocabulary_path)
    # prepare_data(cache_dir, train_data_path, valid_data_path, max_vocabulary_size)