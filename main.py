import tensorflow.compat.v1 as tf

# disabling eager mode
tf.compat.v1.disable_eager_execution()

num_classes = 2
x = tf.placeholder(tf.float32, shape=[ None, 227, 227, 3 ])

y_ = tf.placeholder(tf.float32, [ None, num_classes ])
num_classes = 2
weights = {
    'w1': tf.Variable(tf.random_normal([ 11, 11, 3, 96 ]), name='w1'),  # 11*11*3   96 filters
    'w2': tf.Variable(tf.random_normal([ 5, 5, 96, 256 ]), name='w2'),  # 5*5*96  256 filters

    'w3': tf.Variable(tf.random_normal([ 3, 3, 256, 384 ]), name='w3'),
    'w4': tf.Variable(tf.random_normal([ 3, 3, 384, 384 ]), name='w4'),

    'w5': tf.Variable(tf.random_normal([ 3, 3, 384, 256 ]), name='w5'),

    'wfc1': tf.Variable(tf.random_normal([ 6 * 6 * 256, 4096 ]), name='wfc1'),
    'wfc2': tf.Variable(tf.random_normal([ 4096, 4096 ]), name='wfc2'),

    'wout': tf.Variable(tf.random_normal([ 4096, 2 ]), name='wout')
}

biases = {
    'b1': tf.Variable(tf.random_normal([ 96 ]), name='b1'),
    'b2': tf.Variable(tf.random_normal([ 256 ]), name='b2'),

    'b3': tf.Variable(tf.random_normal([ 384 ]), name='b3'),
    'b4': tf.Variable(tf.random_normal([ 384 ]), name='b4'),

    'b5': tf.Variable(tf.random_normal([ 256 ]), name='b5'),

    'bfc1': tf.Variable(tf.random_normal([ 4096 ]), name='bfc1'),  # 4096
    'bfc2': tf.Variable(tf.random_normal([ 4096 ]), name='bfc2'),  # 4096
    'bout': tf.Variable(tf.random_normal([ num_classes ]), name='bout')  # 2
}


def alex_net(x, weights, biases):
    # reshape input to 227*227*3 size
    x = tf.reshape(x, shape=[ -1, 227, 227, 3 ])

    print("###########################################################################")
    print("size of x is")
    print(x.shape)

    # conv1
    # kernel size=11*11, stride=4

    conv1_in = tf.nn.conv2d(x, weights[ 'w1' ], strides=[ 1, 4, 4, 1 ], padding="SAME")
    conv1_in = tf.nn.bias_add(conv1_in, biases[ 'b1' ])  # y= xw+b
    conv1 = tf.nn.relu(conv1_in)
    # maxpool1   #kernel size 3*3, stride 2

    maxpool1 = tf.nn.max_pool(conv1, ksize=[ 1, 3, 3, 1 ], strides=[ 1, 2, 2, 1 ], padding="VALID")
    print("###########################################################################")
    print("size after 1st conv layer) is ")
    print(maxpool1.shape)

    # conv2
    # kernel 5*5, channels 256, stride 1
    conv2_in = tf.nn.conv2d(maxpool1, weights[ 'w2' ], strides=[ 1, 1, 1, 1 ], padding="SAME")
    conv2_in = tf.nn.bias_add(conv2_in, biases[ 'b2' ])
    conv2 = tf.nn.relu(conv2_in)

    # maxpool2
    # kernel 3*3 stride 2
    maxpool2 = tf.nn.max_pool(conv2, ksize=[ 1, 3, 3, 1 ], strides=[ 1, 2, 2, 1 ], padding="VALID")
    print("###########################################################################")
    print("size after 2nd conv layer) is ")
    print(maxpool2.shape)

    # conv3
    # kernel size=3*3, channels=384, stride 1
    conv3_in = tf.nn.conv2d(maxpool2, weights[ 'w3' ], strides=[ 1, 1, 1, 1 ], padding="SAME")
    conv3_in = tf.nn.bias_add(conv3_in, biases[ 'b3' ])
    conv3 = tf.nn.relu(conv3_in)
    print("###########################################################################")
    print("size after 3rd conv layer) is ")
    print(conv3.shape)

    # conv4
    # kernel=3*3,channels=384,stride=1
    conv4_in = tf.nn.conv2d(conv3, weights[ 'w4' ], strides=[ 1, 1, 1, 1 ], padding="SAME")
    conv4_in = tf.nn.bias_add(conv4_in, biases[ 'b4' ])
    conv4 = tf.nn.relu(conv4_in)
    print("###########################################################################")
    print("size after 4th conv layer) is ")
    print(conv4.shape)

    # conv5
    # kernel 3*3, channels=256, stride 1
    conv5_in = tf.nn.conv2d(conv4, weights[ 'w5' ], strides=[ 1, 1, 1, 1 ], padding="SAME")
    conv5_in = tf.nn.bias_add(conv5_in, biases[ 'b5' ])
    conv5 = tf.nn.relu(conv5_in)

    # maxpool5  # kernel 3*3, stride 2
    maxpool5 = tf.nn.max_pool(conv5, ksize=[ 1, 3, 3, 1 ], strides=[ 1, 2, 2, 1 ], padding="VALID")

    print("###########################################################################")
    print("size after 5th conv layer) is ")
    print(conv5.shape)
    print("###########################################################################")
    print("size after 5th conv layer pooling) is ")
    print(maxpool5.shape)

    fc6 = tf.reshape(maxpool5, [ -1, weights[ 'wfc1' ].get_shape().as_list()[ 0 ] ])
    print("###########################################################################")
    print("size after reshaping the image ) is ")
    print(fc6.shape)

    fc6 = tf.add(tf.matmul(fc6, weights[ 'wfc1' ]), biases[ 'bfc1' ])
    fc6 = tf.nn.relu(fc6)

    # fc7
    fc7 = tf.nn.relu_layer(fc6, weights[ 'wfc2' ], biases[ 'bfc2' ])
    print("###########################################################################")
    print("size after 7th layer) is ")
    print(fc7.shape)

    fc8 = tf.add(tf.matmul(fc7, weights[ 'wout' ]), biases[ 'bout' ])
    out = tf.nn.softmax(fc8)

    return out


# Create the model
model = alex_net(x, weights, biases)
print(model)