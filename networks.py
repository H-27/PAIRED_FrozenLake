import tensorflow as tf

class Direction_Actor_Network(tf.keras.Model):
    def __init__(self, n_actions, map_dims):
        # Environment input branch
        self.env_input = tf.keras.layers.Input(shape=(map_dims[0], map_dims[1], 3))
        self.conv_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.flatten_env = tf.keras.layers.Flatten()

        # Direction input branch
        self.dense_direction = tf.keras.layers.Dense(5, activation='relu')

        # Concatenate environment and direction features
        self.concat_layer = tf.keras.layers.Concatenate()

        # LSTM layer
        self.lstm_layer = tf.keras.layers.LSTM(256)

        # Shared fully connected layers
        self.dense_one = tf.keras.layers.Dense(32, activation='relu')
        self.dense_two = tf.keras.layers.Dense(32, activation='relu')

        # Policy output
        self.policy_output = tf.keras.layers.Dense(n_actions, activation='softmax', name='policy')# Assuming n_actions is the number of possible actions

    def call(self, map, direction, training=True):
        x_env = self.env_input(map)
        x_env = self.conv_layer(x_env)
        x_env = self.flatten_env(x_env)

        x_d = self.direction_input(direction)
        x_d = self.dense_direction(x_d)

        x = self.concat_layer([x_env, x_d])
        x = self.lstm_layer(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        return self.policy_output(x)

class Direction_Critic_Network(tf.keras.Model):
    def __init__(self, n_actions, map_dims):
        # Environment input branch
        self.env_input = tf.keras.layers.Input(shape=(map_dims[0], map_dims[1], 3))
        self.conv_layer = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.flatten_env = tf.keras.layers.Flatten()

        # Direction input branch
        self.dense_direction = tf.keras.layers.Dense(5, activation='relu')

        # Concatenate environment and direction features
        self.concat_layer = tf.keras.layers.Concatenate()

        # LSTM layer
        self.lstm_layer = tf.keras.layers.LSTM(256)

        # Shared fully connected layers
        self.dense_one = tf.keras.layers.Dense(32, activation='relu')
        self.dense_two = tf.keras.layers.Dense(32, activation='relu')

        # Value estimate output
        self.value_output = tf.keras.layers.Dense(1, name='value')

    def call(self, map, direction, training=True):
        x_env = self.env_input(map)
        x_env = self.conv_layer(x_env)
        x_env = self.flatten_env(x_env)

        x_d = self.direction_input(direction)
        x_d = self.dense_direction(x_d)

        x = self.concat_layer([x_env, x_d])
        x = self.lstm_layer(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        return self.value_output(x)


class Conv_Network(tf.keras.Model):

    def __init__(self, n_actions, map_dims, conv_block=False):
        self.use_conv_block = conv_block
        self.shape = (map_dims[0], map_dims[1], 3)
        super(Conv_Network, self).__init__()
        if(conv_block):
            self.layer_conv = Convolution_Block(16, (3, 3), shape=self.shape)
        else:
            self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(32, 3, padding='same', data_format='channels_first') # use default for strides = 1 and padding = 'same' to keep same size
            self.layer_single_conv_two = self.convolution = tf.keras.layers.Conv2D(32, 3, padding='same', data_format='channels_first')
        self.lstm = tf.keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal)#, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal)#, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = n_actions, activation = tf.nn.softmax)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, training=True, transpose=False):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        if(self.use_conv_block):
            x = self.layer_conv(map)
        else:
            x = self.layer_single_conv(map)
            x = self.layer_single_conv_two(x)
        x = self.flatten(x)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x

class Critic_Network(tf.keras.Model):

    def __init__(self, n_actions, map_dims, conv_block=False):
        self.use_conv_block = conv_block
        self.shape = (map_dims[0], map_dims[1], 3)
        super(Critic_Network, self).__init__()
        if(conv_block):
            self.layer_conv = Convolution_Block(16, (3, 3), shape=self.shape)
        else:
            self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(16, 3, padding = 'valid', data_format='channels_first')
            self.dense_extra = tf.keras.layers.Dense(5, activation=tf.nn.relu)
            self.lstm = tf.keras.layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid',)
            self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
            self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
            self.layer_out = tf.keras.layers.Dense(units = 1, activation = None, use_bias=False)
            self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, training=True, transpose=False):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        if(self.use_conv_block):
            x = self.layer_conv(map)
        else:
            x = self.layer_single_conv(map)
        x = self.dense_extra(x)
        x = self.flatten(x)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x

class Convolution_Block(tf.keras.Model):
    # params: number of filters to be used, kernelsize to be used, stridesize to be used with default as 1, padding to be used with default as same, activation function to be used with leaky relu as default, dropoutrate to be used to be used with 0.1 as default
    def __init__(self, n_filters, kernelsize, shape, padding='valid', activation=tf.nn.leaky_relu, dropout=0.1):
        shape = (shape[0], shape[1], 5)
        super(Convolution_Block, self).__init__()
        self.convolution = tf.keras.layers.Conv2D(n_filters, kernelsize, padding= padding)#, data_format='channels_first')#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros, input_shape=shape)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)
        ''' can add performance comparison with dropout layer.'''
        self.dropout = tf.keras.layers.Dropout(dropout)

    @tf.function
    def call(self, input, training=True):
        x = self.convolution(input)
        x = self.batchnorm(x, training)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class Adversary_Network(tf.keras.Model):

    def __init__(self, n_actions, map_dims, conv_block=False):
        self.use_conv_block = conv_block
        self.shape = (map_dims[0], map_dims[1], 3)
        super(Adversary_Network, self).__init__()

        if (conv_block):
            self.layer_conv = Convolution_Block(16, (3, 3), shape=self.shape)
        else:
            self.layer_single_conv = self.convolution = tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='valid', data_format='channels_first')
        self.dense_extra = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.lstm = tf.keras.layers.LSTM(128, activation='tanh', recurrent_activation='sigmoid',)
        self.dense_one = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.dense_two = tf.keras.layers.Dense(32, activation=tf.nn.relu)#, kernel_initializer=tf.initializers.random_normal, bias_initializer=tf.initializers.zeros)
        self.layer_out = tf.keras.layers.Dense(units = n_actions, activation = tf.nn.softmax, use_bias=False)
        self.flatten = tf.keras.layers.Flatten()

    @tf.function
    def call(self, map, timestep, rand_vec, training=True, transpose=False):
        if (transpose):
            # transposes input from channel first to channel last
            map = tf.transpose(map, perm = [0,2,3,1])
        if(self.use_conv_block):
            x = self.layer_conv(map)
        else:
            x = self.layer_single_conv(map)
        x = self.dense_extra(x)
        x = self.flatten(x)
        x = tf.concat([x,timestep], -1)
        x = tf.concat([x,rand_vec], -1)
        x = tf.expand_dims(x, 1)
        x = self.lstm(x)
        x = self.dense_one(x)
        x = self.dense_two(x)
        x = self.layer_out(x)
        return x
