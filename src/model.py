import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator

class Model(object):
    def __init__(self, args, news_title, news_entity, news_group, news_topic, n_user, n_news):
        # n_user = len(user_news)
        # n_news = len(news_entity)
        # n_entity = len(entity_news)
        # n_word = 116603
        n_word = 279215
        self.params = []

        self.use_group = args.use_group
        self.n_filters = args.n_filters
        self.filter_sizes = args.filter_sizes
        self.topic = args.topic
        self.max_session_len = args.session_len
        self.dim = args.dim
        self.lr = args.lr
        self.title_len = args.title_len
        self.batch_size = args.batch_size
        self.news_neighbor = args.news_neighbor
        self.user_neighbor = args.user_neighbor
        self.entity_neighbor = args.entity_neighbor
        self.n_iter = args.n_iter
        self.l2_weight = args.l2_weight
        self.cnn_out_size = args.cnn_out_size
        self.n_topics = args.n_topics

        self.news_group = news_group
        self.news_topic = tf.expand_dims(news_topic,axis=1)
        self.n_user = n_user
        self.n_news = n_news
        self.topic_emb_matrix = tf.get_variable(
            shape=[args.n_topics+1, self.dim], initializer=tf.truncated_normal_initializer(stddev=0.1), name='topic_emb_matrix')
        self.group_embedding = tf.get_variable(name="group_embed", shape=[12, 50],
                                               dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.params.append(self.group_embedding)
        self.params.append(self.topic_emb_matrix)
        self.filter_shape_item = [40, 20, 1, 8]
        self.input_size_item = 10 * 8 * 8
        self.filter_shape_title = [2, 20, 1, 8]
        self.input_size_title = 4 * 8 * 8
        self.filter_shape = [2, 8, 1, 4]
        self.cat_size = 7 * 30 * 4

        self.build_inputs()

        with tf.variable_scope("policy_step"):
            self.cells = []
            for i in range(1):
                lstm_cell = tf.contrib.rnn.LSTMCell(self.dim, use_peepholes=True, state_is_tuple=True)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_rate)
                self.cells.append(lstm_cell)
            self.lstm = tf.contrib.rnn.MultiRNNCell(self.cells, state_is_tuple=True)


        self.title = news_title
        self.news_entity = news_entity
        # self.entity_news = entity_news
        # self.entity_entity = entity_entity
        self.user_mask = tf.constant([[0.] * self.dim] + [[1.] * self.dim] * n_user)
        self.news_mask = tf.constant([[0.] * self.dim] + [[1.] * self.dim] * n_news)
        self.topic_mask = tf.constant([[0.] * self.dim] + [[1.] * self.dim] * args.n_topics)

        self.user_emb_matrix = tf.get_variable(
            shape=[n_user + 1, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='user_emb_matrix')
        self.word_emb_matrix = tf.get_variable(
            shape=[n_word + 1, 50], initializer=tf.truncated_normal_initializer(stddev=0.1), name='word_emb_matrix')
        self.params.append(self.user_emb_matrix)
        self.params.append(self.word_emb_matrix)
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

        self.build_model()
        self.build_train()

    def build_inputs(self):
        self.dropout_rate = tf.placeholder(tf.float32)
        self.user_indices = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='user_indices')
        self.news_indices = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='news_indices')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='labels')
        self.clicks = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_session_len], name='clicked_news')
        self.user_news = tf.placeholder(dtype=tf.int32, shape=[self.n_user, self.news_neighbor], name='user_news')
        self.news_user = tf.placeholder(dtype=tf.int32, shape=[self.n_news, self.user_neighbor], name='user_news')
        self.topic_news = tf.placeholder(dtype=tf.int32, shape=[1+self.n_topics, self.news_neighbor], name='news_topic')

    def build_model(self):
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        self.user_emb_matrix = tf.nn.l2_normalize(self.user_emb_matrix, axis=-1)
        self.topic_emb_matrix = tf.nn.l2_normalize(self.topic_emb_matrix, axis=-1)
        self.word_emb_matrix = tf.nn.l2_normalize(self.word_emb_matrix, axis=-1)
        entities = self.get_neighbors(self.news_indices)
        # [batch_size, dim]
        W1 = tf.get_variable(shape=[self.dim*2, self.dim], initializer=tf.truncated_normal_initializer(stddev=0.1), name='weights1')
        b1 = tf.get_variable(shape=[self.dim], initializer=tf.truncated_normal_initializer(stddev=0.1), name='bias1')
        # self.params.append(W1)

        self.news_embeddings, self.aggregators = self.aggregate(entities)

        # entities = self.get_new_neighbors(self.user_indices, False)
        # self.global_embeddings, aggregators = self.aggregate(entities)
        # self.aggregators.extend(aggregators)
        self.global_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        self.local_embeddings = self.forward()

        self.user_embeddings = tf.matmul(tf.concat([self.global_embeddings,self.local_embeddings], 1), W1)
        # self.user_embeddings = self.global_embeddings




        self.user_embeddings = tf.nn.l2_normalize(self.user_embeddings, axis=-1)
        self.news_embeddings = tf.nn.l2_normalize(self.news_embeddings, axis=-1)
        self.scores = self.classifier_net(self.user_embeddings, self.news_embeddings)
        self.predict_label = tf.cast(tf.argmax(self.scores, 1), tf.int32)


    def build_train(self):
        y_true = tf.one_hot(tf.squeeze(self.labels), 2)
        total_loss = -tf.reduce_sum(y_true * tf.log(tf.nn.softmax(self.scores) + 0.00000000001), -1)
        self.base_loss = tf.reduce_mean(total_loss)

        self.l2_loss = tf.Variable(tf.constant(0., dtype=tf.float32), trainable=False)
        for param in self.params:
            self.l2_loss = tf.add(self.l2_loss, tf.nn.l2_loss(param))

        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def classifier_net(self, x, y):
        with tf.variable_scope("classifier_net", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            input = tf.concat((x, y), -1)
            hidden = tf.layers.dropout(input, self.dropout_rate)
            output = tf.layers.dense(hidden, 2, activation=tf.nn.sigmoid)
        return output

    def attention(self, prev_state, state, time_steps):
        with tf.variable_scope("attention", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("attn_w1", [1, 1, self.dim, self.dim], dtype=tf.float32)
            v = tf.get_variable("attn_v", [self.dim], dtype=tf.float32)
            if w1 not in self.params:
                self.params.append(w1)
            if v not in self.params:
                self.params.append(v)
            hide = tf.reshape(prev_state, [self.batch_size, time_steps, 1, self.dim])
            hidden_features = tf.nn.conv2d(hide, w1, [1, 1, 1, 1], "SAME")
            w2 = tf.get_variable("attn_w2", [self.dim, self.dim], dtype=tf.float32)
            if w2 not in self.params:
                self.params.append(w2)
            y = tf.matmul(state, w2)
            y = tf.reshape(y, [self.batch_size, 1, 1, self.dim])
            s = tf.reduce_sum(v * tf.nn.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            d = tf.reduce_sum(tf.reshape(a, [self.batch_size, time_steps, 1, 1]) * hide, [1, 2])
            attention = tf.reshape(d, [self.batch_size, self.dim])
        return attention



    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)  # [batch_size,1]
        news = [seeds]  # [[batch_size,1]]
        vectors = []
        n = self.news_neighbor
        u = self.user_neighbor
        e = self.entity_neighbor
        print(self.topic)
        hop_vectors = self.convolution(news[0])
        # hop_vectors = tf.nn.l2_normalize(hop_vectors, axis=-1)
        neighbors = tf.concat([tf.nn.embedding_lookup(self.news_user, news[0][:, 0])],axis=1)
        if self.topic:
            neighbors = tf.concat([neighbors, tf.nn.embedding_lookup(self.news_topic, news[0][:, 0])], axis=-1)
        vectors.append(hop_vectors)
        news.append(neighbors)
        print("hop----0", news,vectors)

        if self.n_iter >=1:


            hop_vectors = tf.multiply(tf.nn.embedding_lookup(self.user_emb_matrix, news[1][:, :u]),
                                      tf.nn.embedding_lookup(self.user_mask, news[1][:, :u]))
            neighbors = tf.reshape(tf.gather(self.user_news, news[1][:, :u]), [self.batch_size, -1])
            if self.topic:
                t = tf.multiply(tf.nn.embedding_lookup(self.topic_emb_matrix, news[1][:, u:u + e]),
                                                                tf.nn.embedding_lookup(self.topic_mask, news[1][:, u:u+e]))
                hop_vectors = tf.concat([hop_vectors, t],axis=1)
                neighbors = tf.concat(
                    [neighbors, tf.reshape(tf.gather(self.topic_news, news[1][:, u:u + e]), [self.batch_size, -1])],
                    axis=-1)
            vectors.append(hop_vectors)
            news.append(neighbors)
            print("hop----1", news, vectors)

        if self.n_iter >= 2:
            hop_vectors = self.convolution(news[2])
            neighbors = tf.gather(self.news_user, news[2])
            if self.topic:
                neighbors = tf.concat([neighbors, tf.gather(self.news_topic, news[2])], axis=-1)
            neighbors = tf.reshape(neighbors, [self.batch_size, -1])
            vectors.append(hop_vectors)
            news.append(neighbors)
            print("hop----2", news, vectors)

        if self.n_iter >= 3:
            j = 0
            while j < news[3].shape[1]:
                if j == 0:
                    hop_vectors = tf.multiply(tf.nn.embedding_lookup(self.user_emb_matrix, news[3][:, :u]),
                                              tf.nn.embedding_lookup(self.user_mask, news[3][:, :u]))
                    j += u
                    if self.topic:
                        t = tf.multiply(tf.nn.embedding_lookup(self.topic_emb_matrix, news[3][:, j:j + e]),
                                        tf.nn.embedding_lookup(self.topic_mask, news[3][:, j:j + e]))
                        hop_vectors = tf.concat([hop_vectors, t], axis=1)
                        j += e
                else:
                    t = tf.multiply(tf.nn.embedding_lookup(self.user_emb_matrix, news[3][:, j:j + u]),
                                              tf.nn.embedding_lookup(self.user_mask, news[3][:, j:j + u]))
                    hop_vectors = tf.concat(
                        [hop_vectors, t], axis=1)
                    j += u
                    if self.topic:
                        t = tf.multiply(tf.nn.embedding_lookup(self.topic_emb_matrix, news[3][:, j:j + e]),
                                        tf.nn.embedding_lookup(self.topic_mask, news[3][:, j:j + e]))
                        hop_vectors = tf.concat(
                            [hop_vectors, t], axis=1)
                        j += e
            vectors.append(hop_vectors)
            news.append(neighbors)
            print("hop----3", news, vectors)
        return vectors

    # feature propagation
    def aggregate(self, vectors):
        aggregators = []  # store all aggregators

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.relu)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            # n_neighbor = self.news_neighbor + self.user_neighbor + self.entity_neighbor
            print(vectors)
            for hop in range(self.n_iter - i):
                # shape = [self.batch_size, -1, n_neighbor, self.dim]
                if hop % 2 == 0:
                    if self.topic:
                        shape = [self.batch_size, -1, self.user_neighbor + self.entity_neighbor, self.dim]
                    else:
                        shape = [self.batch_size, -1, self.user_neighbor, self.dim]
                else:
                    shape = [self.batch_size, -1, self.news_neighbor, self.dim]

                print("--hop",hop, vectors[hop], tf.reshape(vectors[hop+1], shape))
                vector = aggregator(self_vectors=vectors[hop],
                                    neighbor_vectors=tf.reshape(vectors[hop+1], shape))
                vector = tf.nn.l2_normalize(vector, axis=-1)
                entity_vectors_next_iter.append(vector)
            vectors = entity_vectors_next_iter

        res = tf.reshape(vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _attention(self,clicked_embeddings,news_embeddings):

        # (batch_size, max_click_history, title_embedding_length)
        clicked_embeddings = tf.reshape(
            clicked_embeddings, shape=[-1, self.max_session_len, self.n_filters * len(self.filter_sizes)])

        # (batch_size, 1, title_embedding_length)
        news_embeddings_expanded = tf.expand_dims(news_embeddings, 1)

        # (batch_size, max_click_history)
        attention_weights = tf.reduce_sum(clicked_embeddings * news_embeddings_expanded, axis=-1)

        # (batch_size, max_click_history)
        attention_weights = tf.nn.softmax(attention_weights, dim=-1)

        # (batch_size, max_click_history, 1)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # (batch_size, title_embedding_length)
        user_embeddings = tf.reduce_sum(clicked_embeddings * attention_weights_expanded, axis=1)

        return user_embeddings

    def convolution(self, inputs):
        title_lookup = tf.reshape(tf.nn.embedding_lookup(self.title, inputs), [-1, self.title_len])
        title_embed = tf.expand_dims(tf.nn.embedding_lookup(self.word_emb_matrix, title_lookup), -1)

        item_lookup = tf.reshape(tf.nn.embedding_lookup(self.news_entity, inputs), [-1, 40])
        group_lookup = tf.reshape(tf.nn.embedding_lookup(self.news_group, inputs), [-1, 40])
        item_embed = tf.expand_dims(tf.nn.embedding_lookup(self.word_emb_matrix, item_lookup), 2)
        group_embed = tf.expand_dims(tf.nn.embedding_lookup(self.group_embedding, group_lookup), 2)
        item_group_embed = tf.expand_dims(
            tf.reshape(tf.concat((item_embed, group_embed), 2), [-1, 80, 50]), -1)

        with tf.variable_scope("conv-maxpool-item-group", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            W_item = tf.get_variable(name='W', shape=self.filter_shape_item, dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_item = tf.get_variable(name='b', shape=[8], dtype=tf.float32)
            if W_item not in self.params:
                self.params.append(W_item)
            if b_item not in self.params:
                self.params.append(b_item)
            conv_item = tf.nn.conv2d(
                item_group_embed,
                W_item,
                strides=[1, 2, 2, 1],
                padding="VALID",
                name="conv")
            h_item = tf.nn.relu(tf.nn.bias_add(conv_item, b_item), name="relu")
            pooled_item = tf.nn.max_pool(
                h_item,
                ksize=[1, 3, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name="pool")
            self.pool_item = tf.reshape(pooled_item, [self.batch_size, -1, self.input_size_item])

        with tf.variable_scope("conv-maxpool-title", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            W_title = tf.get_variable(name='W', shape=self.filter_shape_title, dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b_title = tf.get_variable(name='b', shape=[8], dtype=tf.float32)
            if W_title not in self.params:
                self.params.append(W_title)
            if b_title not in self.params:
                self.params.append(b_title)
            conv_title = tf.nn.conv2d(
                title_embed,
                W_title,
                strides=[1, 2, 2, 1],
                padding="VALID",
                name="conv")
            h_title = tf.nn.relu(tf.nn.bias_add(conv_title, b_title), name="relu")
            pooled_title = tf.nn.max_pool(
                h_title,
                ksize=[1, 2, 1, 1],
                strides=[1, 1, 2, 1],
                padding='VALID',
                name="pool")
            pool_title = tf.reshape(pooled_title, [self.batch_size, -1, self.input_size_title])

        pooled = tf.concat((self.pool_item, pool_title), -1)
        pool = tf.layers.dense(pooled, self.cnn_out_size, activation=tf.nn.relu)
        # W_ = tf.get_variable(name='W_', shape=[self.dim*2,self.dim], dtype=tf.float32,
        #                          initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # if W_ not in self.params:
        #     self.params.append(W_)
        # pool = tf.nn.relu(tf.matmul(pooled,W_))
        # hidden = tf.layers.dropout(pooled, self.dropout_rate)
        # pool = tf.layers.dense(hidden, self.cnn_out_size, activation=tf.nn.relu)
        return pool

    def convolution_net(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        with tf.variable_scope("conv-maxpool-net", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name='W', shape=self.filter_shape, dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable(name='b', shape=[4], dtype=tf.float32)
            if W not in self.params:
                self.params.append(W)
            if b not in self.params:
                self.params.append(b)
            conv = tf.nn.conv2d(
                inputs,
                W,
                strides=[1, 1, 2, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, 3, 3, 1],
                strides=[1, 1, 2, 1],
                padding='VALID',
                name="pool")
            pool = tf.reshape(pooled, [self.batch_size, self.cat_size])
            return pool

    def full_mlp(self, input):
        with tf.variable_scope("MLP_for_full", reuse=tf.AUTO_REUSE):
            hidden = tf.layers.dropout(input, self.dropout_rate)
            output = tf.layers.dense(hidden, self.cnn_out_size, activation=tf.nn.sigmoid)
        return output

    def attention_net(self, prev_state, state, time_steps):
        with tf.variable_scope("attention_net", initializer=tf.truncated_normal_initializer(stddev=0.1),
                               reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("attn_w1", [1, 1, self.cnn_out_size, self.cnn_out_size], dtype=tf.float32)
            v = tf.get_variable("attn_v", [self.cnn_out_size], dtype=tf.float32)
            if w1 not in self.params:
                self.params.append(w1)
            if v not in self.params:
                self.params.append(v)
            hide = tf.reshape(prev_state, [self.batch_size, time_steps, 1, self.cnn_out_size])
            hidden_features = tf.nn.conv2d(hide, w1, [1, 1, 1, 1], "SAME")
            w2 = tf.get_variable("attn_w2", [self.cnn_out_size, self.cnn_out_size], dtype=tf.float32)
            if w2 not in self.params:
                self.params.append(w2)
            y = tf.matmul(state, w2)
            y = tf.reshape(y, [self.batch_size, 1, 1, self.cnn_out_size])
            s = tf.reduce_sum(v * tf.nn.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            d = tf.reduce_sum(tf.reshape(a, [self.batch_size, time_steps, 1, 1]) * hide, [1, 2])
            attention = tf.reshape(d, [self.batch_size, self.cnn_out_size])
        return attention

    def step(self, prev_state, state, inputs, time_steps):
        prev_embedding = tf.squeeze(self.convolution(inputs))
        output, new_state = self.lstm(prev_embedding, state)
        h = new_state[-1][-1]
        atten = self.attention(prev_state, h, time_steps)
        current_state = tf.concat((prev_state, tf.expand_dims(atten, 1)), 1)
        return atten, new_state, current_state

    def forward(self):
        clicks_embedding = self.convolution(self.clicks)
        candidate_embedding = tf.squeeze(self.convolution(self.news_indices))
        feature1 = self.attention_net(clicks_embedding, candidate_embedding, self.max_session_len)
        inputs = tf.split(self.clicks, self.max_session_len, 1)
        embedding = tf.squeeze(self.convolution(inputs[0]))
        init_state = self.lstm.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        h, state = self.lstm(embedding, init_state)
        prev_state = tf.expand_dims(h, 1)
        for time_steps in range(self.max_session_len - 1):
            h, state, prev_state = self.step(prev_state, state, inputs[time_steps + 1], time_steps + 1)
        feature2 = self.convolution_net(prev_state)
        concate = tf.concat((feature2, feature1), -1)
        out = self.full_mlp(concate)
        # output = self.classifier_net(out, candidate_embedding)
        return out

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss,self.news_embeddings,self.user_embeddings], feed_dict)

    def eval(self, sess, feed_dict):
        labels, predict = sess.run([self.labels,self.predict_label], feed_dict)

        # labels, predict = sess.run([self.labels, self.scores], feed_dict)
        auc = roc_auc_score(labels, predict)
        f1 = f1_score(labels, predict)
        p = precision_score(labels, predict)
        r = recall_score(labels, predict)
        return auc, f1, p,r
