import tensorflow as tf

class Actor():
    def __init__(self, act_size, state_size, batch_size, sess):
        
        self.input_size = state_size
        self.output_size = act_size
        self.sess = sess
        self.batch_size = batch_size

        self.input = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

        # create actor net
        self.W_1 = tf.Variable(tf.truncated_normal([self.input_size, 400], stddev=0.01))
        self.b_1 = tf.Variable(tf.zeros(400))

        self.W_2 = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_2 = tf.Variable(tf.zeros(300))

        self.W_3 = tf.Variable(tf.truncated_normal([300, self.output_size],stddev=0.003))
        self.b_3 = tf.Variable(tf.zeros(self.output_size))

        self.network_params = tf.trainable_variables()

        self.out_1 = tf.nn.relu(tf.matmul(self.input, self.W_1) + self.b_1)
        self.out_2 = tf.nn.relu(tf.matmul(self.out_1, self.W_2) + self.b_2)
        self.out = tf.nn.tanh(tf.matmul(self.out_2, self.W_3) + self.b_3)

        # create target actor net
        self.W_1_t = tf.Variable(tf.truncated_normal([self.input_size, 400], stddev=0.01))
        self.b_1_t = tf.Variable(tf.zeros(400))

        self.W_2_t = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_2_t = tf.Variable(tf.zeros(300))

        self.W_3_t = tf.Variable(tf.truncated_normal([300, self.output_size],stddev=0.003))
        self.b_3_t = tf.Variable(tf.zeros(self.output_size))

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.out_1_t = tf.nn.relu(tf.matmul(self.input, self.W_1_t) + self.b_1_t)
        self.out_2_t = tf.nn.relu(tf.matmul(self.out_1_t, self.W_2_t) + self.b_2_t)
        self.out_t = tf.nn.tanh(tf.matmul(self.out_2_t, self.W_3_t) + self.b_3_t)
        
        # update the weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], 0.001) +
                                                  tf.multiply(self.target_network_params[i], 1. - 0.001))
                for i in range(len(self.target_network_params))]

        # gradients
        self.action_grad = tf.placeholder(tf.float32, [None, self.output_size])
        # self.actor_grads = tf.gradients(self.out,self.network_params,-self.action_grad)

        self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_grad)
        self.actor_grads = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        #per la backprop
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(zip(self.actor_grads, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def predict(self, state):
        return self.sess.run(self.out, feed_dict={self.input: state})

    def target_predict(self, state):
        return self.sess.run(self.out_t, feed_dict={self.input: state})

    def train(self, states, gradient):
        self.sess.run(self.optimize, feed_dict={self.input: states, self.action_grad: gradient})

    def target_train(self):
        self.sess.run(self.update_target_network_params)