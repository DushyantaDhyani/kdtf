import tensorflow as tf
import os


class BigModel:
    def __init__(self, args, model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.dropoutprob = args.dropoutprob
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "bigmodel"
        self.temperature = args.temperature
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + ".ckpt")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type

        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="%s_%s" % (self.model_type, "wc1")),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name="%s_%s" % (self.model_type, "wc2")),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name="%s_%s" % (self.model_type, "wd1")),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.num_classes]), name="%s_%s" % (self.model_type, "out"))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32]), name="%s_%s" % (self.model_type, "bc1")),
            'bc2': tf.Variable(tf.random_normal([64]), name="%s_%s" % (self.model_type, "bc2")),
            'bd1': tf.Variable(tf.random_normal([1024]), name="%s_%s" % (self.model_type, "bd1")),
            'out': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "out"))
        }

        self.build_model()
        self.saver = tf.train.Saver()

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        with tf.name_scope("%sconv2d" % (self.model_type)), tf.variable_scope("%sconv2d" % (self.model_type)):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        with tf.name_scope("%smaxpool2d" % (self.model_type)), tf.variable_scope("%smaxpool2d" % (self.model_type)):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                  padding='SAME')

    # Create model
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))
        self.keep_prob = tf.placeholder(tf.float32,
                                        name="%s_%s" % (self.model_type, "dropoutprob"))  # dropout (keep probability)
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemp"))
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        with tf.name_scope("%sinputreshape" % (self.model_type)), tf.variable_scope(
                        "%sinputreshape" % (self.model_type)):
            x = tf.reshape(self.X, shape=[-1, 28, 28, 1])

        # Convolution Layer
        with tf.name_scope("%sconvmaxpool" % (self.model_type)), tf.variable_scope("%sconvmaxpool" % (self.model_type)):
            conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
            # Max Pooling (down-sampling)
            conv1 = self.maxpool2d(conv1, k=2)

            # Convolution Layer
            conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            # Max Pooling (down-sampling)
            conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, self.keep_prob)

            # Output, class prediction
            logits = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out']) / self.softmax_temperature

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                        "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss", self.loss_op)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, dataset):

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())

        print("Starting Training")

        train_data = dataset.get_train_data()
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        max_accuracy = 0

        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = train_data.next_batch(self.batch_size)
            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                       feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.dropoutprob,
                                                  self.softmax_temperature: self.temperature})
            if (step % self.display_step) == 0 or step == 1:
                # Calculate Validation loss and accuracy
                validation_x, validation_y = dataset.get_validation_data()
                loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.X: validation_x,
                                                                                    self.Y: validation_y,
                                                                                    self.keep_prob: 1.0,
                                                                                    self.softmax_temperature: 1.0})

                if acc > max_accuracy:
                    save_path = self.saver.save(self.sess, self.checkpoint_path)
                    print("Model Checkpointed to %s " % (save_path))
                    max_accuracy = acc

                print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                    loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))
        else:
            # Final Evaluation and checkpointing before training ends
            validation_x, validation_y = dataset.get_validation_data()
            loss, acc = self.sess.run([self.loss_op, self.accuracy], feed_dict={self.X: validation_x,
                                                                                self.Y: validation_y,
                                                                                self.keep_prob: 1.0,
                                                                                self.softmax_temperature: 1.0})

            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model Checkpointed to %s " % (save_path))

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.keep_prob: 1.0, self.softmax_temperature: temperature})

    def run_inference(self, dataset):
        test_images, test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           self.keep_prob: 1.0,
                                                                           self.softmax_temperature: 1.0
                                                                           }))

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


class SmallModel:
    def __init__(self, args, model_type):
        self.learning_rate = 0.001
        self.num_steps = args.num_steps
        self.batch_size = args.batch_size
        self.display_step = args.display_step
        self.n_hidden_1 = 256  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.num_input = 784  # MNIST data input (img shape: 28*28)
        self.num_classes = 10
        self.temperature = args.temperature
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_file = "smallmodel"
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        self.max_checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_file + "max")
        self.log_dir = os.path.join(args.log_dir, self.checkpoint_file)
        self.model_type = model_type

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1]),
                              name="%s_%s" % (self.model_type, "h1")),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]),
                              name="%s_%s" % (self.model_type, "h2")),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]),
                               name="%s_%s" % (self.model_type, "out")),
            'linear': tf.Variable(tf.random_normal([self.num_input, self.num_classes]),
                                  name="%s_%s" % (self.model_type, "linear"))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]), name="%s_%s" % (self.model_type, "b1")),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]), name="%s_%s" % (self.model_type, "b2")),
            'out': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "out")),
            'linear': tf.Variable(tf.random_normal([self.num_classes]), name="%s_%s" % (self.model_type, "linear"))
        }

        self.build_model()

        self.saver = tf.train.Saver()

    # Create model
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.num_input], name="%s_%s" % (self.model_type, "xinput"))
        self.Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "yinput"))

        self.flag = tf.placeholder(tf.bool, None, name="%s_%s" % (self.model_type, "flag"))
        self.soft_Y = tf.placeholder(tf.float32, [None, self.num_classes], name="%s_%s" % (self.model_type, "softy"))
        self.softmax_temperature = tf.placeholder(tf.float32, name="%s_%s" % (self.model_type, "softmaxtemperature"))

        with tf.name_scope("%sfclayer" % (self.model_type)), tf.variable_scope("%sfclayer" % (self.model_type)):
            # Hidden fully connected layer with 256 neurons
            # layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
            # # Hidden fully connected layer with 256 neurons
            # layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            # # Output fully connected layer with a neuron for each class
            # logits = (tf.matmul(layer_2, self.weights['out']) + self.biases['out'])
            logits = tf.add(tf.matmul(self.X, self.weights['linear']), self.biases['linear'])

        with tf.name_scope("%sprediction" % (self.model_type)), tf.variable_scope("%sprediction" % (self.model_type)):
            self.prediction = tf.nn.softmax(logits)

            self.correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        with tf.name_scope("%soptimization" % (self.model_type)), tf.variable_scope(
                        "%soptimization" % (self.model_type)):
            # Define loss and optimizer
            self.loss_op_standard = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.Y))

            self.total_loss = self.loss_op_standard

            self.loss_op_soft = tf.cond(self.flag,
                                        true_fn=lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                            logits=logits / self.softmax_temperature, labels=self.soft_Y)),
                                        false_fn=lambda: 0.0)

            self.total_loss += tf.square(self.softmax_temperature) * self.loss_op_soft

            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(0.05)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss)

        with tf.name_scope("%ssummarization" % (self.model_type)), tf.variable_scope(
                        "%ssummarization" % (self.model_type)):
            tf.summary.scalar("loss_op_standard", self.loss_op_standard)
            tf.summary.scalar("total_loss", self.total_loss)
            # Create a summary to monitor accuracy tensor
            tf.summary.scalar("accuracy", self.accuracy)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

            # Merge all summaries into a single op

            # If using TF 1.6 or above, simply use the following merge_all function
            # which supports scoping
            # self.merged_summary_op = tf.summary.merge_all(scope=self.model_type)

            # Explicitly using scoping for TF versions below 1.6

            def mymergingfunction(scope_str):
                with tf.name_scope("%s_%s" % (self.model_type, "summarymerger")), tf.variable_scope(
                                "%s_%s" % (self.model_type, "summarymerger")):
                    from tensorflow.python.framework import ops as _ops
                    key = _ops.GraphKeys.SUMMARIES
                    summary_ops = _ops.get_collection(key, scope=scope_str)
                    if not summary_ops:
                        return None
                    else:
                        return tf.summary.merge(summary_ops)

            self.merged_summary_op = mymergingfunction(self.model_type)

    def start_session(self):
        self.sess = tf.Session()

    def close_session(self):
        self.sess.close()

    def train(self, dataset, teacher_model=None):
        teacher_flag = False
        if teacher_model is not None:
            teacher_flag = True

        # Initialize the variables (i.e. assign their default value)
        self.sess.run(tf.global_variables_initializer())
        train_data = dataset.get_train_data()
        train_summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        max_accuracy = 0

        print("Starting Training")

        def dev_step():
            validation_x, validation_y = dataset.get_validation_data()
            loss, acc = self.sess.run([self.loss_op_standard, self.accuracy], feed_dict={self.X: validation_x,
                                                                                         self.Y: validation_y,
                                                                                         # self.soft_Y: validation_y,
                                                                                         self.flag: False,
                                                                                         self.softmax_temperature: 1.0})

            if acc > max_accuracy:
                save_path = self.saver.save(self.sess, self.checkpoint_path)
                print("Model Checkpointed to %s " % (save_path))

            print("Step " + str(step) + ", Validation Loss= " + "{:.4f}".format(
                loss) + ", Validation Accuracy= " + "{:.3f}".format(acc))
            return max(acc, max_accuracy)

        for step in range(1, self.num_steps + 1):
            batch_x, batch_y = train_data.next_batch(self.batch_size)
            soft_targets = batch_y
            if teacher_flag:
                soft_targets = teacher_model.predict(batch_x, self.temperature)

            # self.sess.run(self.train_op,
            _, summary = self.sess.run([self.train_op, self.merged_summary_op],
                                       feed_dict={self.X: batch_x,
                                                  self.Y: batch_y,
                                                  self.soft_Y: soft_targets,
                                                  self.flag: teacher_flag,
                                                  self.softmax_temperature: self.temperature}
                                       )
            if (step % self.display_step) == 0 or step == 1:
                max_accuracy = dev_step()
        else:
            # Final Evaluation and checkpointing before training ends
            dev_step()

        train_summary_writer.close()

        print("Optimization Finished!")

    def predict(self, data_X, temperature=1.0):
        return self.sess.run(self.prediction,
                             feed_dict={self.X: data_X, self.flag: False, self.softmax_temperature: temperature})

    def run_inference(self, dataset):
        test_images, test_labels = dataset.get_test_data()
        print("Testing Accuracy:", self.sess.run(self.accuracy, feed_dict={self.X: test_images,
                                                                           self.Y: test_labels,
                                                                           # self.soft_Y: test_labels,
                                                                           self.flag: False,
                                                                           self.softmax_temperature: 1.0
                                                                           }))

    def load_model_from_file(self, load_path):
        ckpt = tf.train.get_checkpoint_state(load_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
