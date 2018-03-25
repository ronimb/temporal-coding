import numpy as np
import tensorflow as tf


class Tempotron():
    def __init__(self, threshold_value, learning_rate,
                 input_shape=(10, 500), weights=None, reduce_func='sum'):
        # Handling input
        if len(np.shape(input_shape)) == 0:  # This is for future use when input shapes are generalized
            input_shape = (input_shape,)
        self.threshold_value = threshold_value
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.weights = weights

        # Model constants
        self.thresh = tf.constant(threshold_value, dtype=tf.float32)
        self.lmbda = tf.constant(learning_rate, dtype=tf.float32)

        # Instantiating the model
        self.reduce_func = getattr(tf, 'reduce_{}'.format(reduce_func))  # Setting reduce function for weight updates
        self.setup_model()
        self.setup_training()
        self.start_session()

    def start_session(self):
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def initvals(self):
        tf.global_variables_initializer().run()

    def setup_model(self):
        # Input layer
        self.x = tf.placeholder(tf.float32, shape=(None, *self.input_shape))

        # Configurable weights - Must manually define weights with proper shape!
        if self.weights == None:
            # Weight initialization # Currently fixed for N x T shapes
            self.w = tf.Variable(
                tf.random_uniform(shape=(1, self.input_shape[0]), minval=0, maxval=1))
        elif type(self.weights) == tf.Variable:
            self.w = self.weights
        else:
            self.w = tf.Variable(self.weights)

        # Activation layer - NOTE: CURRENTLY NOT GENERALIZED TO ANY SHAPE!
        ## NEED TO SWITCH TO A VERSION WITH NO SQUEEZE!
        self.h = tf.tensordot(self.w, self.x, [[1], [1]])

        # Thresholding activation
        self.h_t = tf.greater_equal(self.h, self.thresh)

        # Decision node
        self.y = tf.reduce_any(self.h_t, axis=2)

        # Real labels
        self.y_ = tf.placeholder(tf.bool)

        # Correctly classified samples
        self.correct = tf.equal(self.y, self.y_)

    def setup_training(self):
        ### This should return a self.train_step object for use in the training
        ### phase, currently fixed method

        # Filter for only wrong trials
        self.wrong_trials = tf.where(~self.correct)[:, 1]

        self.xfilt = tf.gather(self.x, self.wrong_trials, axis=0)
        self.hfilt = tf.gather(self.h, self.wrong_trials, axis=1)

        # Select parameters for variable update
        self.strongest_node = tf.argmax(self.hfilt, axis=2)

        self.inds = tf.stack(
            [tf.range(tf.size(self.strongest_node[0, :], out_type=tf.int64)), self.strongest_node[0, :]],
            axis=1)
        self.x_upd = tf.gather_nd(tf.transpose(self.xfilt, [0, 2, 1]), self.inds)

        self.y__upd = tf.cast(tf.gather(self.y_, self.wrong_trials), self.x_upd.dtype)[:, np.newaxis]

        # weight update
        self.w_upd = self.reduce_func(self.lmbda * (2 * self.y__upd - 1) * self.x_upd, 0)

        self.train_step = tf.assign(self.w, tf.add(self.w, self.w_upd))

    def train(self, data, labels, iter_num=1000, batch_size=None, train_mode='batch',
              return_params=False, verbose=False):
        '''
        The tempotron training function,
        This function takes data and real labels as input, and trains the model using these samples.

        has two alternative 'train_mode' options:
        'batch' : Each training step will run 'batch_size' samples in parallel, 'iter_num' such steps take place
        'single' : Each training step runs on precisely one sample, 'iter_num' repetitions over all samples take place
        '''

        global param_list

        def make_dict(pnames, plist, indlist):
            d = {n: [] for n in pnames}
            d['batch_inds'] = []
            for row, inds in zip(plist, indlist):
                for n, i in zip(pnames, range(len(pnames))):
                    d[n].append(row[i])
                d['batch_inds'].append(inds)
            d['w'] = np.subtract(np.squeeze(d['w']), d['w_upd'])
            d['correct'] = np.mean(d['correct'], 2)
            return d

        if return_params:
            pnames = ['w', 'correct', 'w_upd']
            var_list = [getattr(self, n) for n in pnames]
            param_list = []

        if train_mode == 'batch':
            if batch_size == None:
                # All iterations use all of the samples
                for i in range(iter_num):
                    fd = {self.x: data, self.y_: labels}
                    if return_params:
                        param_list.append(self.sess.run([*var_list, self.train_step], fd))
                    else:
                        self.sess.run(self.train_step, fd)
                if return_params:
                    params = make_dict(pnames, param_list, indlist=iter_num * [np.arange(data.shape[0])])
                    params['w_post'] = [p[-1] for p in param_list]

            else:
                # Each iteration gets a random subset with size=batch_size
                all_inds = []
                for i in range(iter_num):
                    # This is techincally problematic because of sampling with replacement in index selection
                    inp_inds = np.random.randint(data.shape[0], size=batch_size, dtype=int)
                    fd = {self.x: data[inp_inds], self.y_: labels[inp_inds]}
                    if return_params:
                        all_inds.append(inp_inds)
                        param_list.append(self.sess.run([*var_list, self.train_step], fd))
                    else:
                        self.sess.run(self.train_step, fd)
                if return_params:
                    params = make_dict(pnames, param_list, indlist=all_inds)
                    params['w_post'] = [p[-1] for p in param_list]

        elif train_mode == 'single':
            all_inds = []
            for i in range(iter_num):
                j = 0
                for d, l in zip(data, labels):
                    j += 1
                    fd = {self.x: d[np.newaxis,], self.y_: l[np.newaxis,]}
                    if return_params:
                        all_inds.append([j])
                        param_list.append(self.sess.run([*var_list, self.train_step], fd))
                    else:
                        self.sess.run(self.train_step, fd)
            if return_params:
                params = make_dict(pnames, param_list, indlist=all_inds)
                params['w_post'] = [p[-1] for p in param_list]

        if return_params:
            return params

    def accuracy(self, data, labels):
        fd = {self.x: data, self.y_: labels}
        c = self.correct.eval(fd)
        return c.sum() / c.size