# -*- coding: utf-8 -*-

import os

import RNN_utils
from config import Config
from RNN_model import BiRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
conf = Config()
wav_files, text_labels = RNN_utils.get_wavs_lables()
words_size, words, word_num_map = RNN_utils.create_dict(text_labels)
bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
bi_rnn.build_train()
