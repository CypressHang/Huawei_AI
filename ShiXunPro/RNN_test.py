# -*- coding: utf-8 -*-

import os

import RNN_utils
from config import Config
from RNN_model import BiRNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

conf = Config()

wav_files, text_labels = RNN_utils.get_wavs_lables()

words_size, words, word_num_map = RNN_utils.create_dict(text_labels)

# bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
# bi_rnn.build_test()

wav_files = ['input.wav']
txt_labels = ['北京 丰台区 农民 自己 花钱 筹办 万 佛 延寿 寺 迎春 庙会 吸引 了 区内 六十 支 秧歌队 参赛']
bi_rnn = BiRNN(wav_files, text_labels, words_size, words, word_num_map)
bi_rnn.build_target_wav_file_test(wav_files, txt_labels)
