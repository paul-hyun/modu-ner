# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, parameter):
        self.parameter = parameter

    def build_model(self):
        self._build_placeholder()

        # 포지셔널 인코딩 행렬 생성
        position_encode = self.positional_encoding(self.parameter['word_embedding_size'], self.parameter['sentence_length'])

        # { "morph": 0, "morph_tag": 1, "tag" : 2, "character": 3, .. }
        self._embedding_matrix = [] # shape word: (331273, 16), shape char: (2176, 16)
        for item in self.parameter["embedding"]:
            self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))

        # 각각의 임베딩 값을 가져온다
        self._embeddings = []
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph) + position_encode) # shape (batch, 180, 16)
        # self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character)) # shape (batch, 180, 8, 16)

        encoder_outputs = self.encoder(self._embeddings[0], self.parameter['word_embedding_size'], self.parameter['word_embedding_size'], self.parameter['word_embedding_size'], 3)
        decoder_outputs = self.decoder(self.ne_dict, encoder_outputs, 15, 15, 15, 3)

        # 단어 수 만큼 차원을 변환
        logits = tf.keras.layers.Dense(15)(decoder_outputs)
        # 예측 시퀀스를 내놓음
        self.predict = tf.argmax(logits, 2)

        labels = tf.one_hot(self.label, 15)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        # accuracy = tf.metrics.accuracy(labels=self.label, predictions=predict, name='accOp')

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, global_step=self.global_step)

    """ placeholder 생성 """
    def _build_placeholder(self):
        self.morph = tf.placeholder(tf.int32, [None, None]) # shape (batch, 180)
        self.ne_dict = tf.placeholder(tf.float32, [None, int(self.parameter["sentence_length"]), int(self.parameter["n_class"] / 2)]) # shape (batch, 180, 15)
        self.character = tf.placeholder(tf.int32, [None, None, None]) # shape (batch, 180, 8)
        self.dropout_rate = tf.placeholder(tf.float32)
        self.sequence = tf.placeholder(tf.int32, [None]) # shape (batch,)
        self.character_len = tf.placeholder(tf.int32, [None, None]) # shape (batch, 180)
        self.label = tf.placeholder(tf.int32, [None, None]) # shape (batch, 180)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.Variable(0.02, trainable=False, name='learning_rate')

    """ embedding 백터 생성 """
    def _build_embedding(self, n_tokens, dimention, name="embedding"):
        embedding_weights = tf.get_variable(
            name, [n_tokens, dimention],
            dtype=tf.float32,
        )
        return embedding_weights
    
    """ 내적연산 Attention with Mask """
    def scaled_dot_product_attention(self, query, key, value, masked=False):
        key_seq_length = float(key.get_shape().as_list()[-2])
        key = tf.transpose(key, perm=[0, 2, 1]) # 0차원에 대한 전치행렬
        outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # mask 값 구함 (현재값 기준으로 과거 값만 이용 가능하도록 함)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1) # padding은 음수 엄청 큰 값
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # 0: padding, 아니면: outputs

        attention_map = tf.nn.softmax(outputs)

        return tf.matmul(attention_map, value)
    
    """ 어텐션 맵을 어렷 만들어 다양한 특징에 대한 어텐션을 볼 수 있도록 한 방법 """
    def multi_head_attention(self, query, key, value, num_units, heads, masked=False):
        # 입력한 query, key, value에 대해 리니어 레이어를 거치도록 함
        query = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(query)
        key = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(key)
        value = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(value)

        # 선형층을 통과한 query, key, value에 대해서 지정된 헤드 수 만큼 입력하도록 피쳐들을 분리
        query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
        key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
        value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

        # 셀프 어텐션 연산
        attention_feature = self.scaled_dot_product_attention(query, key, value, masked)
        # 다시 나눠진 헤드에 대한 피쳐들을 하나로 다시 모음
        attn_outputs = tf.concat(tf.split(attention_feature, heads, axis=0), axis=-1)
        # 리니어 레이어를 거침
        attn_outputs = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(attn_outputs)

        return attn_outputs
    
    """ 포지션-와이즈 피드포워드 네트워크 """
    def feed_forward(self, inputs, num_units):
        # 출력 디멘젼은 입력 디멘젼과 동일해야 한다.
        feature_shape = inputs.get_shape()[-1]
        # 활성화 함수는 첫 레이어에서만 적용을 한다.
        inner_layer = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dense(feature_shape)(inner_layer)

        return outputs

    """ sublayer_connection이라 """
    def sublayer_connection(self, inputs, sublayer, dropout=0.2):
        outputs = self.layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
        return outputs

    """ 레이어 노말라이제이션 """
    def layer_norm(self, inputs, eps=1e-6):
        feature_shape = inputs.get_shape()[-1:]
        #  평균과 표준편차을 넘겨 준다.
        mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
        std = tf.keras.backend.std(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
        gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

        return gamma * (inputs - mean) / (std + eps) + beta

    """ 인코더 모듈은 멀티 헤드 어텐션 레이어와 피드포워드 네트워크로 한 레이어를 구성 """
    def encoder_module(self, inputs, model_dim, ffn_dim, heads):
        # 셀프 어텐션 레이어
        self_attn = self.sublayer_connection(inputs, self.multi_head_attention(inputs, inputs, inputs, model_dim, heads))
        # 포지션 와이즈 피드포워드 레이어
        outputs = self.sublayer_connection(self_attn, self.feed_forward(self_attn, ffn_dim))

        return outputs
    
    """ 인코더가 여러번 반복되는 형태 """
    def encoder(self, inputs, model_dim, ffn_dim, heads, num_layers):
        outputs = inputs
        for i in range(num_layers):
            outputs = self.encoder_module(outputs, model_dim, ffn_dim, heads)

        return outputs

    """ 두개의 어텐션과 하나의 피드포워드 레이어로 구성 """
    def decoder_module(self, inputs, encoder_outputs, model_dim, ffn_dim, heads):
        # 마스크 셀프 어텐션 레이어
        masked_self_attn = self.sublayer_connection(inputs, self.multi_head_attention(inputs, inputs, inputs, model_dim, heads, masked=True))
        # 인코더 디코더 임베딩을 위한 셀프 어텐션 레이어
        self_attn = self.sublayer_connection(masked_self_attn, self.multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, model_dim, heads))
        # 포지션 와이즈 피드포워드 레이어
        outputs = self.sublayer_connection(self_attn, self.feed_forward(self_attn, ffn_dim))

        return outputs
    
    """ 블록을 여러번 반복하는 전체 디코더 모듈 """
    def decoder(self, inputs, encoder_outputs, model_dim, ffn_dim, heads, num_layers):
        outputs = inputs
        for i in range(num_layers):
            outputs = self.decoder_module(outputs, encoder_outputs, model_dim, ffn_dim, heads)

        return outputs

    """ 입력 시퀀스 정보에 대한 순서 정보를 부가적으로 주입 """
    def positional_encoding(self, dim, sentence_length):
        encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                                for pos in range(sentence_length) for i in range(dim)])

        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])

        return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


if __name__ == "__main__":
    parameter = {}
    model = Model(parameter)

    inputs = tf.Variable([[[1,2,3,4,5,6,7,8,9,0,1,2,3,4,5], [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5], [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5]]], dtype=tf.float32)
    # output = model.multi_head_attention(inputs, inputs, inputs, 16, 16)
    # output = model.encoder_module(inputs, 16, 16, 16)
    encoder_outputs = model.encoder(inputs, 15, 15, 15, 2)
    # output = model.decoder_module(inputs, encoder_outputs, 16, 16, 16)
    output = model.decoder(inputs, encoder_outputs, 15, 15, 15, 2)
    # output = model.positional_encoding(3, 100)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', output)

    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    trace = sess.run(inputs, feed_dict={})
    print(trace.shape, trace)
    # trace = sess.run(model.trace, feed_dict={})
    # print(trace.shape, trace)
    trace = sess.run(output, feed_dict={})
    print(trace.shape, trace)
