import tensorflow as tf
import utils
import data
import models.modelbase as base
import time

class MultiLayerLSTM(base.ModelBase):
    def __init__(self, modelconfig, char2vec=None, output_char2vec=None, input=None, seq_length=100, type="multi"):
        base.ModelBase.__init__(self, modelconfig, char2vec, output_char2vec, input)
        self.seq_length = seq_length
        self.type = type
        tf.reset_default_graph()

    def run(self):
        if type == "multi":
            print('****** MultiLayer LSTM Initialize ******')
        elif type == "bimul":
            print('****** Bidirectional LSTM Initialize ******')

        training_dataset, valid_dataset = data.make_sequences(self.input, self.char2vec, self.output_char2vec, self.seq_length)

        input_batch = training_dataset.input_batch
        target_batch = training_dataset.target_batch
        seq_lens = training_dataset.seq_lens

        hidden_size = self.modelconfig.hidden_size

        X = tf.placeholder(tf.int32, [None, self.seq_length])  # X data
        X_onehot = tf.one_hot(X, self.modelconfig.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

        Y = tf.placeholder(tf.int32, [None, self.seq_length])  # Y label

        Seqlen = tf.placeholder(tf.int32, [None])

        keep_prob = tf.placeholder(tf.float32)

        if self.type != "multi":
            with tf.variable_scope('cell_def'):
                cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=keep_prob)
                cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
                cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=keep_prob)
                multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

            with tf.variable_scope('rnn_def'):
                outputs, _states = tf.nn.dynamic_rnn(
                    multi_cell, X_onehot,  dtype=tf.float32, sequence_length=Seqlen)

        elif self.type != "bimul":
            with tf.variable_scope('cell_def'):
                forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
                forward = tf.nn.rnn_cell.DropoutWrapper(forward, output_keep_prob=keep_prob)
                backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
                backward = tf.nn.rnn_cell.DropoutWrapper(backward, output_keep_prob=keep_prob)

            with tf.variable_scope('rnn_def'):
                outputs, states = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs=X_onehot, dtype=tf.float32, sequence_length=Seqlen)
                outputs = tf.concat(values=outputs, axis=2)

        model = tf.layers.dense(outputs, self.modelconfig.output_size, activation=None)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
        prediction = tf.argmax(model, axis=2)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        eval = utils.Evaluation(self.type, self.modelconfig.epoch, 25)

        print('------------ Training ------------ ')
        last_time = time.time()
        for epoch in range(self.modelconfig.epoch):

            _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch,
                                                             Y: target_batch,
                                                             Seqlen: seq_lens,
                                                             keep_prob: 0.8})
            if epoch % 25 == 24:
                result = sess.run(prediction, feed_dict={X: valid_dataset.input_batch,
                                                         Y: valid_dataset.target_batch,
                                                         Seqlen: valid_dataset.seq_lens,
                                                         keep_prob: 1})
                accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(Y, tf.int64)), tf.float32))
                accuracy_ret = sess.run(accuracy, feed_dict={Y: valid_dataset.target_batch})
                speed = time.time() - last_time
                print('Epoch:', '%04d  ' % (epoch + 1),
                      'accuracy =', '{:.6f}  '.format(accuracy_ret),
                      'cost =', '{:.6f}'.format(loss),
                      'speed =', '{:.2f}'.format(speed), 'sec')
                last_time = time.time()

                avg_p, avg_r, avg_f = utils.print_evaluation(valid_dataset.target_batch, result, self.output_char2vec.char_dict)
                eval.set(accuracy_ret, loss, speed, avg_p, avg_r, avg_f)
                print('')
        eval_dict = eval.get_avg()

        print('------------ Testing ------------ ')
        test_sentences = data.read_data("data/test/BHXX0035.txt", 30)
        test_dataset, _ = data.make_sequences(test_sentences, self.char2vec, self.output_char2vec, self.seq_length, make_valid=False)

        result = sess.run(prediction, feed_dict={X: test_dataset.input_batch,
                                                      Y: test_dataset.target_batch,
                                                      Seqlen: test_dataset.seq_lens,
                                                      keep_prob: 1})

        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(Y, tf.int64)), tf.float32))
        accuracy_ret = sess.run(accuracy, feed_dict={Y: test_dataset.target_batch})

        print('Accuracy =', '{:.6f}'.format(accuracy_ret))

        for index, predict_sequence in enumerate(result):
            target_output, prediction_output = data.compare_sentence(self.output_char2vec,
                                                                     test_dataset.target_batch[index],
                                                                     test_dataset.input_source[index],
                                                                     predict_sequence)
            print("target sentence:    ", target_output[1])
            print("prediction sentence:", prediction_output[1])