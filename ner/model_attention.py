# -*- coding: utf-8 -*-

import tensorflow as tf

class Model:
    def __init__(self, parameter):
        self.parameter = parameter
        self.num_heads = 4
        self.num_units = 2 * self.parameter["lstm_units"] # 32 # 280

    def build_model(self):
        self._build_placeholder()

        # { "morph": 0, "morph_tag": 1, "tag" : 2, "character": 3, .. }
        self._embedding_matrix = []
        for item in self.parameter["embedding"]:
            self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))

        # 각각의 임베딩 값을 가져온다
        self._embeddings = []
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character))

        # 음절을 이용한 임베딩 값을 구한다.
        character_embedding = tf.reshape(self._embeddings[1], [-1, self.parameter["word_length"], self.parameter["embedding"][1][2]])
        char_len = tf.reshape(self.character_len, [-1])

        character_emb_rnn, _, _ = self._build_birnn_model(character_embedding, char_len, self.parameter["char_lstm_units"], self.dropout_rate, last=True, scope="char_layer")

        # 위에서 구한 모든 임베딩 값을 concat 한다.
        all_data_emb = self.ne_dict
        for i in range(0, len(self._embeddings)-1):
            all_data_emb = tf.concat([all_data_emb, self._embeddings[i]], axis=2)
        all_data_emb = tf.concat([all_data_emb, character_emb_rnn], axis=2)

        # 모든 데이터를 가져와서 Bi-RNN 실시
        sentence_output, W, B = self._build_birnn_model(all_data_emb, self.sequence, self.parameter["lstm_units"], self.dropout_rate, scope="all_data_layer")
        sentence_output = tf.matmul(sentence_output, W) + B

        # 마지막으로 CRF 를 실시 한다
        crf_cost, crf_weight, crf_bias = self._build_crf_layer(sentence_output)

        self.train_op = self._build_output_layer(crf_cost)
        self.cost = crf_cost

    def _build_placeholder(self):
        self.morph = tf.placeholder(tf.int32, [None, None])
        self.ne_dict = tf.placeholder(tf.float32, [None, None, int(self.parameter["n_class"] / 2)])
        self.character = tf.placeholder(tf.int32, [None, None, None])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.sequence = tf.placeholder(tf.int32, [None])
        self.character_len = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_embedding(self, n_tokens, dimention, name="embedding"):
        embedding_weights = tf.get_variable(
            name, [n_tokens, dimention],
            dtype=tf.float32,
        )
        return embedding_weights

    def _build_single_cell(self, lstm_units, keep_prob):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

        return cell

    def _build_weight(self, shape, scope="weight"):
        with tf.variable_scope(scope):
            W = tf.get_variable(name="W", shape=[shape[0], shape[1]], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b", shape=[shape[1]], dtype=tf.float32, initializer = tf.zeros_initializer())
        return W, b

    def _build_birnn_model(self, target, seq_len, lstm_units, keep_prob, last=False, scope="layer"):
        with tf.variable_scope("forward_" + scope):
            lstm_fw_cell = self._build_single_cell(lstm_units, keep_prob)

        with tf.variable_scope("backward_" + scope):
            lstm_bw_cell = self._build_single_cell(lstm_units, keep_prob)

        with tf.variable_scope("birnn-lstm_" + scope):
            _output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, dtype=tf.float32,
                                                        inputs = target, sequence_length = seq_len, scope="rnn_" + scope)
            if last:
                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                # outputs = self.self_attention(outputs)
                outputs = tf.reshape(outputs, shape=[-1, self.parameter["sentence_length"], 2 * lstm_units])
            else:
                (output_fw, output_bw), _ = _output
                outputs = tf.concat([output_fw, output_bw], axis=2)
                outputs = self.self_attention(outputs)
                outputs = tf.reshape(outputs, shape=[-1, 2 * lstm_units])

            W, b = self._build_weight([2 * self.parameter["lstm_units"], self.parameter["n_class"]], scope="output" + scope)
        return outputs, W, b

    def _build_crf_layer(self, target):
        with tf.variable_scope("crf_layer"):
            W, B = self._build_weight([self.parameter["n_class"], self.parameter["n_class"]], scope="weight_bias")
            matricized_unary_scores = tf.matmul(target, W) + B
            matricized_unary_scores = tf.reshape(matricized_unary_scores, [-1, self.parameter["sentence_length"], self.parameter["n_class"]])
    
            self.matricized_unary_scores = matricized_unary_scores
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.matricized_unary_scores, self.label, self.sequence)
            cost = tf.reduce_mean(-self.log_likelihood)

            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(matricized_unary_scores, self.transition_params, self.sequence)

        return cost, W, B

    def _build_output_layer(self, cost):
        with tf.variable_scope("output_layer"):
            train_op = tf.train.AdamOptimizer(self.parameter["learning_rate"]).minimize(cost, global_step=self.global_step)
        return train_op
    
    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta

        return outputs

    def self_attention(self, keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            if self.parameter["mode"] == "train":
                outputs = tf.nn.dropout(outputs, keep_prob=self.dropout_rate)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs


if __name__ == "__main__":
    parameter = {"embedding" : {
                    "morph" : [ 10, 10 ],
                    "morph_tag" : [ 10, 10 ],
                    "tag" : [ 10, 10 ],
                    "ne_dict" : [ 10, 10 ],
                    "character" : [ 10, 10 ],
                    }, "lstm_units" : 32, "keep_prob" : 0.65,
                    "sequence_length": 300, "n_class" : 100, "batch_size": 128,
                    "learning_rate" : 0.002
                }
    model = Model(parameter)
    model.build_model()
