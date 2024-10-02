import tensorflow as tf

class Actor_Network(tf.keras.Model):

    def __init__(self, n_actions):
        super(Actor_Network, self).__init__()
        self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', data_format='channels_first')
        self.dense_direction = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = n_actions, activation = tf.nn.softmax, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, position, direction, use_direction = True, use_position = True, training=True, transpose=True):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        x = self.layer_single_conv(map)
        x = self.flatten(x)
        if (use_direction):
            x_d = self.dense_direction(direction)
            x = tf.concat([x, x_d], -1)
        if (use_position):
            x = tf.concat([x, position], -1)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x

class Critic_Network(tf.keras.Model):

    def __init__(self):
        super(Critic_Network, self).__init__()
        self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='same', data_format='channels_first')
        self.dense_direction = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = 1, activation = None, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, position, direction,  use_direction = True, use_position = True, training=True, transpose=True):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        x = self.layer_single_conv(map)
        x = self.flatten(x)
        if (use_direction):
            x_d = self.dense_direction(direction)
            x = tf.concat([x, x_d], -1)
        if (use_position):
            x = tf.concat([x, position], -1)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x


class Conv_Network(tf.keras.Model):

    def __init__(self, n_actions, map_dims):
        self.shape = (map_dims[0], map_dims[1], 3)
        super(Conv_Network, self).__init__()
        self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(32, 3, padding='same', data_format='channels_first') # use default for strides = 1 and padding = 'same' to keep same size
        self.lstm = tf.keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal)#, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal)#, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = n_actions, activation = tf.nn.softmax)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, training=True, transpose=True):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        x = self.layer_single_conv(map)
        x = self.flatten(x)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x

class Adversary_Network(tf.keras.Model):

    def __init__(self, n_actions, map_dims):
        self.shape = (map_dims[0], map_dims[1], 3)
        super(Adversary_Network, self).__init__()

        self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='valid', data_format='channels_first')
        self.dense_extra = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = n_actions, activation = tf.nn.softmax, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, timestep, rand_vec, training=True, transpose=True):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        x = self.layer_single_conv(map)
        x_t = self.dense_extra(timestep)
        x = self.flatten(x)
        x = tf.concat([x,x_t], -1)
        x = tf.concat([x,rand_vec], -1)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x
