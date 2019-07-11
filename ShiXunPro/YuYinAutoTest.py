# encoding: utf-8
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from collections import Counter
import platform as plat
from depends.ShengXueModel import ModelSpeech
from depends.YuYanModel import ModelLanguage



# 获取文件夹下所有的WAV文件
def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # print(filename)
                filename_path = os.path.join(dirpath, filename)
                # print(filename_path)
                wav_files.append(filename_path)
    return wav_files


# 获取wav文件对应的翻译文字
def get_tran_texts(wav_files, tran_path):
    tran_texts = []
    for wav_file in wav_files:
        (wav_path, wav_filename) = os.path.split(wav_file)
        tran_file = os.path.join(tran_path, wav_filename + '.trn')
        # print(tran_file)
        if os.path.exists(tran_file) is False:
            return None

        fd = open(tran_file, 'r',encoding='utf8')
        text = fd.readline()
        tran_texts.append(text.split('\n')[0])
        fd.close()
    return tran_texts


# 获取wav和对应的翻译文字
def get_wav_files_and_tran_texts(wav_path, tran_path):
    wav_files = get_wav_files(wav_path)
    tran_texts = get_tran_texts(wav_files, tran_path)

    return wav_files, tran_texts


# 旧的训练集使用该方法获取音频文件名和译文
def get_wavs_lables(wav_path, label_file):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
                    continue
                wav_files.append(filename_path)

    labels_dict = {}
    with open(label_file, 'rb') as f:
        for label in f:
            label = label.strip(b'\n')
            label_id = label.split(b' ', 1)[0]
            label_text = label.split(b' ', 1)[1]
            labels_dict[label_id.decode('ascii')] = label_text.decode('utf-8')

    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]

        if wav_id in labels_dict:
            labels.append(labels_dict[wav_id])
            new_wav_files.append(wav_file)

    return new_wav_files, labels


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


# 将稀疏矩阵的字向量转成文字
# tuple是sparse_tuple_from函数的返回值
def sparse_tuple_to_texts_ch(tuple, words):
    # 索引
    indices = tuple[0]
    # 字向量
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        c = ' ' if c == SPACE_INDEX else words[c]
        results[index] = results[index] + c

    return results


# 将密集矩阵的字向量转成文字
def ndarray_to_text_ch(value, words):
    results = ''
    for i in range(len(value)):
        results += words[value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


# 创建序列的稀疏表示
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


# 将音频数据转为时间序列（列）和MFCC（行）的矩阵，将对应的译文转成字向量
def get_audio_and_transcriptch(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files != None:
        txt_labels = txt_files

    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # load audio and convert to features
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')
        # print(word_num_map)
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))

        # load text transcription and convert to numerical array
        target = []
        if txt_files != None:  # txt_obj是文件
            target = get_ch_lable_v(txt_obj, word_num_map)
        else:
            target = get_ch_lable_v(None, word_num_map, txt_obj)  # txt_obj是labels
        # target = text_to_char_array(target)
        transcript.append(target)
        transcript_len.append(len(target))

    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len


# 将字符转成向量，其实就是根据字找到字在word_num_map中所应对的下标
def get_ch_lable_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)

    to_num = lambda word: word_num_map.get(word, words_size)

    if txt_file != None:
        txt_label = get_ch_lable(txt_file)

    # print(txt_label)
    labels_vector = list(map(to_num, txt_label))
    # print(labels_vector)
    return labels_vector


def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            # labels =label.decode('utf-8')
            labels = labels + label.decode('gb2312')
            # labels.append(label.decode('gb2312'))

    return labels


# 将音频信息转成MFCC特征
# 参数说明---audio_filename：音频文件   numcep：梅尔倒谱系数个数
#       numcontext：对于每个时间段，要包含的上下文样本个数
def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    # 加载音频文件
    fs, audio = wav.read(audio_filename)
    # 获取MFCC系数
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)
    # 打印MFCC系数的形状，得到比如(955, 26)的形状
    # 955表示时间序列，26表示每个序列的MFCC的特征值为26个
    # 这个形状因文件而异，不同文件可能有不同长度的时间序列，但是，每个序列的特征值数量都是一样的
    # print(np.shape(orig_inputs))

    # 因为我们使用双向循环神经网络来训练,它的输出包含正、反向的结
    # 果,相当于每一个时间序列都扩大了一倍,所以
    # 为了保证总时序不变,使用orig_inputs =
    # orig_inputs[::2]对orig_inputs每隔一行进行一次
    # 取样。这样被忽略的那个序列可以用后文中反向
    # RNN生成的输出来代替,维持了总的序列长度。
    orig_inputs = orig_inputs[::2]  # (478, 26)
    # print(np.shape(orig_inputs))
    # 因为我们讲解和实际使用的numcontext=9，所以下面的备注我都以numcontext=9来讲解
    # 这里装的就是我们要返回的数据，因为同时要考虑前9个和后9个时间序列，
    # 所以每个时间序列组合了19*26=494个MFCC特征数
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))
    # print(np.shape(train_inputs))#)(478, 494)

    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    # time_slices保存的是时间切片，也就是有多少个时间序列
    time_slices = range(train_inputs.shape[0])

    # context_past_min和context_future_max用来计算哪些序列需要补零
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext

    # 开始遍历所有序列
    for time_slice in time_slices:
        # 对前9个时间序列的MFCC特征补0，不需要补零的，则直接获取前9个时间序列的特征
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert (len(empty_source_past) + len(data_source_past) == numcontext)

        # 对后9个时间序列的MFCC特征补0，不需要补零的，则直接获取后9个时间序列的特征
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert (len(empty_source_future) + len(data_source_future) == numcontext)

        # 前9个时间序列的特征
        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        # 后9个时间序列的特征
        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        # 将前9个时间序列和当前时间序列以及后9个时间序列组合
        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert (len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)

    # 将数据使用正太分布标准化，减去均值然后再除以方差
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    return train_inputs


# 对齐处理
def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    # [478 512 503 406 481 509 422 465]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)

    # maxlen，该批次中，最长的序列长度
    if maxlen is None:
        maxlen = np.max(lengths)

    # 在下面的主循环中，从第一个非空序列中获取样本形状以检查一致性
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # 序列为空，跳过

        # post表示后补零，pre表示前补零
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    return x, lengths

#路径
# wav_path = 'data/data_thchs30/train'  #这个太大了 换个小的
wav_path = 'test/'
label_file = 'data/data_thchs30/data'
# wav_files, labels = get_wavs_lables(wav_path,label_file)
wav_files, labels = get_wav_files_and_tran_texts(wav_path, label_file)

# 字表
all_words = []
for label in labels:
    # print(label)
    all_words += [word for word in label]
counter = Counter(all_words)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))

print('字表大小:', words_size)

# 梅尔倒谱系数的个数
n_input = 26
# 对于每个时间序列，要包含上下文样本的个数
n_context = 9
# batch大小
batch_size = 8


def next_batch(wav_files, labels, start_idx=0, batch_size=1):
    filesize = len(labels)
    # 计算要获取的序列的开始和结束下标
    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)
    # 获取要训练的音频文件路径和对于的译文
    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    # 将音频文件转成要训练的数据
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(None,
                                                                             wav_files,
                                                                             n_input,
                                                                             n_context, word_num_map, txt_labels)

    start_idx += batch_size
    # Verify that the start_idx is not largVerify that the start_idx is not ler than total available sample size
    if start_idx >= filesize:
        start_idx = -1

    # Pad input to max_time_step of this batch
    # 如果多个文件将长度统一，支持按最大截断或补0
    source, source_lengths = pad_sequences(source)
    # 返回序列的稀疏表示
    sparse_labels = sparse_tuple_from(target)

    return start_idx, source, source_lengths, sparse_labels


def get_speech_file(wav_file, labels):
    # 获取要训练的音频文件路径和对于的译文
    txt_labels = [labels[0]]
    wav_files = [wav_file]
    # 将音频文件转成要训练的数据
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(None,
                                                                             wav_files,
                                                                             n_input,
                                                                             n_context, word_num_map, txt_labels)

    # Pad input to max_time_step of this batch
    # 如果多个文件将长度统一，支持按最大截断或补0
    source, source_lengths = pad_sequences(source)
    # 返回序列的稀疏表示
    sparse_labels = sparse_tuple_from(target)

    return source, source_lengths, sparse_labels




def variable_on_cpu(name, shape, initializer):
    # Use the /cpu:0 device for scoped operations 原来是GPU  但是用不了
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN_model(batch_x, seq_length, n_input, n_context, n_character, keep_dropout):
    b_stddev = 0.046875
    h_stddev = 0.046875

    n_hidden = 1024
    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_hidden_5 = 1024
    n_cell_dim = 1024
    n_hidden_3 = 2 * 1024

    keep_dropout_rate = 0.95
    relu_clip = 20

    """
    used to create a variable in CPU memory.
    """
    # batch_x_shape: [batch_size, amax_stepsize, n_input + 2 * n_input * n_context]
    batch_x_shape = tf.shape(batch_x)

    # 将输入转成时间序列优先
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # 再转成2维传入第一层
    # [amax_stepsize * batch_size, n_input + 2 * n_input * n_context]
    batch_x = tf.reshape(batch_x, [-1, n_input + 2 * n_input * n_context])

    # 使用clipped RELU activation and dropout.
    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)

    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)

    # 双向rnn
    with tf.name_scope('lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=keep_dropout)
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=keep_dropout)

        # `layer_3`  `[amax_stepsize, batch_size, 2 * n_cell_dim]`
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        # 连接正反向结果[amax_stepsize, batch_size, 2 * n_cell_dim]
        outputs = tf.concat(outputs, 2)

        # to a single tensor of shape [amax_stepsize * batch_size, 2 * n_cell_dim]
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b_stddev))
        h5 = variable_on_cpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)

    with tf.name_scope('fc6'):
        # 全连接层用于softmax分类
        b6 = variable_on_cpu('b6', [n_character], tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_cpu('h6', [n_hidden_5, n_character], tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
    print('layer_6', layer_6)
    print('batch_x_shape[0]', batch_x_shape[0])
    # 将2维[amax_stepsize * batch_size, n_character]转成3维 time-major [amax_stepsize, batch_size, n_character].
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_character])
    print('layer_6', layer_6)

    print('n_character:' + str(n_character))
    # exit()
    # Output shape: [amax_stepsize, batch_size, n_character]
    return layer_6



def TargetTest(path='input.wav'):
    datapath = '/'
    modelpath = 'saver/Model/model_speech/'
    ms = ModelSpeech(datapath)
    ms.LoadModel(modelpath + 'speech_model251_e_0_step_12000.model')
    # ms.TestModel(datapath, str_dataset='test', data_count = 64, out_report = True)
    r = ms.RecognizeSpeech_FromFile(path)

    ml = ModelLanguage('model_language')
    ml.LoadModel()

    str_pinyin = r
    r = ml.SpeechToText(str_pinyin)
    return r



def CheckpointTest():
    # input_tensor为输入音频数据，由前面分析可知，它的结构是[batch_size, amax_stepsize, n_input + (2 * n_input * n_context)]
    # 其中，batch_size是batch的长度，amax_stepsize是时序长度，n_input + (2 * n_input * n_context)是MFCC特征数，
    # batch_size是可变的，所以设为None，由于每一批次的时序长度不固定，所有，amax_stepsize也设为None
    input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)], name='input')
    # Use sparse_placeholder; will generate a SparseTensor, required by ctc_loss op.
    # targets保存的是音频数据对应的文本的系数张量，所以用sparse_placeholder创建一个稀疏张量
    targets = tf.sparse_placeholder(tf.int32, name='targets')
    # seq_length保存的是当前batch数据的时序长度
    seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
    # keep_dropout则是dropout的参数
    keep_dropout = tf.placeholder(tf.float32)

    # logits is the non-normalized output/activations from the last layer.
    # logits will be input for the loss function.
    # nn_model is from the import statement in the load_model function
    logits = BiRNN_model(input_tensor, tf.to_int64(seq_length), n_input, n_context, words_size + 1, keep_dropout)

    aa = ctc_ops.ctc_loss(targets, logits, seq_length)
    # 使用ctc loss计算损失
    avg_loss = tf.reduce_mean(aa)

    # 优化器
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)

    # 使用CTC decoder
    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_greedy_decoder(logits, seq_length, merge_repeated=True)

    # 计算编辑距离
    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
        # 计算label error rate (accuracy)
        ler = tf.reduce_mean(distance, name='label_error_rate')

    # 迭代次数
    epochs = 150
    # 模型保存地址
    savedir = "saver/"
    # 如果该目录不存在，新建
    if os.path.exists(savedir) == False:
        os.mkdir(savedir)

    # 生成saver
    saver = tf.train.Saver(max_to_keep=1)
    # 创建session
    with tf.Session() as sess:
        # 初始化
        sess.run(tf.global_variables_initializer())
        # 没有模型的话，就重新初始化
        kpt = tf.train.latest_checkpoint(savedir)
        print("kpt:", kpt)
        startepo = 0
        if kpt != None:
            saver.restore(sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])

        # 要识别的语音文件
        wav_file = 'input.wav'

        source, source_lengths, sparse_labels = get_speech_file(wav_file, labels)
        feed2 = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths, keep_dropout: 1.0}
        d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
        dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
        if (len(dense_decoded) > 0):
            decoded_str = ndarray_to_text_ch(dense_decoded[0], words)
            print('Decoded:  {}'.format(decoded_str))


def AutoTest():
    print('开始自动测试：\n')
    countTrue = 0
    count=0
    for index in range(len(wav_files)):
        print('音频文件:  ' + wav_files[index])
        print('文字内容:  ' + labels[index])
        result=TargetTest(wav_files[index])
        print('识别的内容:',result)
        print('\n')

        stand=labels[index].replace(' ','')

        count = count+len(stand)
        for i in range(count-1):
            if(i>= len(result) or i>=len(stand)):
                break
            if(result[i] == stand[i]):
                countTrue = countTrue + 1.8

    print('识别的内容正确率为:', countTrue/count)


AutoTest()