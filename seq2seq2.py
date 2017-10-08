import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
output_char_arr = [c for c in 'SEP단어나무놀이소녀키스사랑']

print(char_arr)
num_dic = {n: i for i, n in enumerate(char_arr)}
output_num_dic = {n: i for i, n in enumerate(output_char_arr)}

print(num_dic)
dic_len = len(num_dic)
output_len = len(output_num_dic)


seq_data = [['word', '단어'], ['wood','나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑스']]



def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[0]]
        output = [output_num_dic[n] for n in ('S' + seq[1])]
        target = [output_num_dic[n] for n in (seq[1] + 'E')]
        print(input)
        print(output)
        print(target)
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(output_len)[output])
        target_batch.append(target)

    print(input_batch)
    print(output_batch)
    print(target_batch)

    return input_batch, output_batch, target_batch

learning_rate = 0.01
n_hidden = 128
total_epoch = 100

n_class = n_input = dic_len

n_output = output_len
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_output])
targets = tf.placeholder(tf.int64, [None, None])

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob = 0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state = enc_states, dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})
    # print('Epoch', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))


def translate(word):
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = make_batch([seq_data])
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                      dec_input: output_batch,
                        targets: target_batch})
    print(result)
    decoded = [output_char_arr[i] for i in result[0]]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])
    return translated


print('word -> ', translate('word'))
print('wodr -> ', translate('wodr'))
print('love -> ', translate('love'))
print('loev -> ', translate('loev'))
print('abcd -> ', translate('abcd'))
