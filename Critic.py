import tensorflow as tf

class Critic():
    def __init__(self, act_size, state_size, num_act_vars, sess):
        self.input_size = state_size
        self.act_size = act_size
        self.sess = sess

        self.input = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None, self.act_size], dtype=tf.float32)

        # create critic net
        self.W_1 = tf.Variable(tf.truncated_normal([self.input_size, 400],stddev=0.01))
        self.b_1 = tf.Variable(tf.zeros(400))

        self.W_2 = tf.Variable(tf.truncated_normal([self.act_size, 300],stddev=0.01))
        self.b_2 = tf.Variable(tf.zeros(300))

        self.W_3 = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_3 = tf.Variable(tf.zeros(300))

        self.W_4 = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_4 = tf.Variable(tf.zeros(1))


        self.out_1 = tf.nn.relu(tf.matmul(self.input, self.W_1) + self.b_1)
        self.out_2 = tf.nn.relu(tf.matmul(self.out_1,self.W_3) + tf.matmul(self.action, self.W_2) + self.b_2)
        self.out = tf.matmul(self.out_2, self.W_4) + self.b_4

        self.network_params = tf.trainable_variables()[num_act_vars:]

        # create target critic net
        self.W_1_t = tf.Variable(tf.truncated_normal([self.input_size, 400],stddev=0.01))
        self.b_1_t = tf.Variable(tf.zeros(400))

        self.W_2_t = tf.Variable(tf.truncated_normal([self.act_size, 300],stddev=0.01))
        self.b_2_t = tf.Variable(tf.zeros(300))

        self.W_3_t = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_3_t = tf.Variable(tf.zeros(300))

        self.W_4_t = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_4_t = tf.Variable(tf.zeros(1))


        self.out_1_t = tf.nn.relu(tf.matmul(self.input, self.W_1_t) + self.b_1_t)
        self.out_2_t = tf.nn.relu(tf.matmul(self.out_1_t,self.W_3_t) + tf.matmul(self.action,self.W_2_t) + self.b_2_t)
        self.out_t = tf.matmul(self.out_2_t, self.W_4_t) + self.b_4_t

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_act_vars):]

        # update target net
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], 0.001) +
                                                  tf.multiply(self.target_network_params[i], 1. - 0.001))
                for i in range(len(self.target_network_params))]
        
        # define the loss and train
        self.loss = tf.reduce_mean(tf.square(self.target - self.out))
        self.optimize = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.action_grads = tf.gradients(self.out, self.action)

    def predict(self, state, action):
        return self.sess.run(self.out, feed_dict={self.input: state, self.action:action})
    
    def target_predict(self, state, action):
        return self.sess.run(self.out_t, feed_dict={self.input: state, self.action:action})

    def gradient(self, state, action):
        return self.sess.run(self.action_grads, feed_dict={self.input: state, self.action: action})

    def train(self, state, action, target):
        self.sess.run(self.optimize, feed_dict={self.input: state, self.action: action, self.target: target})

    def target_train(self):
        self.sess.run(self.update_target_network_params)