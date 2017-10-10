import tensorflow as tf
import numpy as np
import utils
import data

def make_sequences(input_data, char2vec, output_char2vec, seq_length, make_valid=True):
    total_frames = 1 if (len(input_data) < seq_length) else int(len(input_data) / seq_length)

    training_dataset = None
    valid_dataset = None

    if make_valid == True:
        frames_training = int(total_frames * 0.9)
        test_case = ['training', 'validation']
    else:
        frames_training = total_frames
        test_case = ['training']

    for case in test_case:
        input_batch = []
        input_source = []
        target_batch = []
        seqlens = []

        if case == 'training':
            start = 0
            end = frames_training
        else:
            start = frames_training + 1
            end = total_frames

        for i in range(start, end):
            input_str, output_str = data.extract_punc(input_data[i * seq_length: (i + 1) * seq_length],
                                                      char2vec.char_dict, output_char2vec.char_dict)
            # print(i, input_str, '->', output_str)
            input_source.append(input_str)
            x = []
            for ch in input_str:
                if ch in char2vec.char_dict:
                    x.append(char2vec.char_dict[ch])
                else:
                    x.append(char2vec.char_dict['<unk>'])
            y = [output_char2vec.char_dict[c] for c in output_str]  # y str to index

            seqlens.append(len(x))

            if len(x) != seq_length:
                diff = seq_length - len(x)
                for _ in range(diff):
                    x.append(0)
                    input_str.append(' ')
            if len(y) != seq_length:
                diff = seq_length - len(y)
                for _ in range(diff):
                    y.append(0)
            # print(y)
            input_batch.append(x)
            target_batch.append(y)

        if case == 'training':
            training_dataset = DataSet(input_batch, input_source, target_batch, seqlens)
        else:
            valid_dataset = DataSet(input_batch, input_source, target_batch, seqlens)
    return training_dataset, valid_dataset

def make_weight_mat(input_batch, seq_lens):
    weight_mat = []
    for i in range(len(input_batch)):
        weight_list = []
        # Todo: need refactoring
        for j in range(30):
            if j < seq_lens[i]:
                weight_list.append(1)
            else:
                weight_list.append(0)
        weight_mat.append(weight_list)
    return weight_mat

def compare_sentence(output_char2vec, target, input_source, prediction, printable):
    result_output = ([output_char2vec.r_char_dict[c] for c in prediction])
    target_output = ([output_char2vec.r_char_dict[c] for c in target])

    if printable == True:
        result_str = data.apply_punc("".join(input_source), result_output)
        target_str = data.apply_punc("".join(input_source), target_output)

        print("Target:    ", target_str)
        print("Prediction:", result_str)
    return result_output, target_output


class DataSet():
    def __init__(self, input_batch, input_source, target_batch, seq_lens):
        self.input_batch = input_batch
        self.input_source = input_source
        self.target_batch = target_batch
        self.seq_lens = seq_lens

class ModelConfiguration():
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, layers=1, bi=False):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.layers = layers
        self.bi_mul = 2 if bi else 1

class ModelBase():
    def __init__(self, model, char2vec=None, output_char2vec=None, input=None):
        print('****** Model Basic Initialize ******')
        self.model = model
        if char2vec is None:
            self.char2vec = Char2Vec()
        else:
            self.char2vec = char2vec

        if output_char2vec is None:
            self.output_char2vec = self.char2vec
        else:
            self.output_char2vec = output_char2vec

        self.input = input

class MultiLayerLSTM(ModelBase):
    def __init__(self, model, char2vec=None, output_char2vec=None, input=None, seq_length=30):
        print('****** MultiLayer LSTM Initialize ******')
        ModelBase.__init__(self, model, char2vec, output_char2vec, input)
        self.prediction = None
        self.sess = None
        self.seq_length = seq_length


    def run(self):
        training_dataset, valid_dataset = make_sequences(self.input, self.char2vec, self.output_char2vec, self.seq_length)

        input_batch = training_dataset.input_batch
        target_batch = training_dataset.target_batch
        seq_lens = training_dataset.seq_lens

        hidden_size = self.model.hidden_size
        batch_size = len(input_batch)

        tf.reset_default_graph()
        X = tf.placeholder(tf.int32, [None, self.seq_length])  # X data
        Y = tf.placeholder(tf.int32, [None, self.seq_length])  # Y label

        Seqlen = tf.placeholder(tf.int32, [None])

        Weight = tf.placeholder(tf.float32, [None, None])  # Weight
        # X_one_hot = tf.one_hot(X, hidden_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
        X_one_hot = tf.one_hot(X, self.model.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

        model = None
        # print(X_one_hot)
        if self.model.bi_mul != 2:
            with tf.variable_scope('cell_def'):
                cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
                # cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
                cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

                multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
            with tf.variable_scope('rnn_def'):
                outputs, _states = tf.nn.dynamic_rnn(
                    multi_cell, X_one_hot,  dtype=tf.float32, sequence_length=Seqlen)
            print(outputs)
            model = tf.layers.dense(outputs, self.model.output_size, activation=None)

        else:
            forward = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            backward = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs=X_one_hot, dtype=tf.float32, sequence_length=Seqlen)
            print(outputs[-1].dtype.base_dtype)
            model = tf.layers.dense(outputs[-1], self.model.output_size, activation=None )


        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        weights = make_weight_mat(input_batch, seq_lens)

        prediction = tf.argmax(model, axis=2) # axis 2 ??
        target = tf.cast(Y, tf.int64) #Y #tf.argmax(Y, axis=1) # axis 2 ??

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for epoch in range(1000):
            l, loss = sess.run([optimizer, cost], feed_dict={X:input_batch, Y:target_batch, Weight:weights, Seqlen:seq_lens})

            if epoch%25 == 24:
                print("epoch #", epoch)

                valid_weights = make_weight_mat(valid_dataset.input_batch, valid_dataset.seq_lens)

                result = sess.run(prediction, feed_dict={X: valid_dataset.input_batch, Y:valid_dataset.target_batch, Weight:valid_weights, Seqlen:valid_dataset.seq_lens})
                # print(result)
                sess.run(target, feed_dict={Y: valid_dataset.target_batch})
                is_correct = tf.equal(prediction, target)

                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                print(sess.run(accuracy, feed_dict={X: valid_dataset.input_batch, Y:valid_dataset.target_batch, Weight:valid_weights, Seqlen:valid_dataset.seq_lens}))

                result_output_mat = []
                target_output_mat = []
                # print(result)
                for index, sq in enumerate(result):
                    result_output, target_output = compare_sentence(self.output_char2vec, valid_dataset.target_batch[index], valid_dataset.input_source[index], sq, True if epoch == 999 else False)
                    result_output_mat.append(result_output)
                    target_output_mat.append(target_output)

                utils.print_pc_matrix(result_output_mat, target_output_mat)

        test_sentence = list("예를 들어, 이와 같은 경우 불법이 확실하다.")
        test_dataset, _ = make_sequences(test_sentence, self.char2vec, self.output_char2vec, self.seq_length, make_valid=False)
        weights = make_weight_mat(test_dataset.input_batch, test_dataset.seq_lens)

        # print(test_dataset.input_batch)
        # print(test_dataset.target_batch)

        result = sess.run(prediction, feed_dict={X: test_dataset.input_batch, Y: test_dataset.target_batch,
                                                           Weight: weights, Seqlen: test_dataset.seq_lens})
        # print(result)

        for index, sq in enumerate(result):
            result_output, target_output = compare_sentence(output_char2vec, test_dataset.target_batch[index],
                                                                 test_dataset.input_source[index], sq, printable=True)


