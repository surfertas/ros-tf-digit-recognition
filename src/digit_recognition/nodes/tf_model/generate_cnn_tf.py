import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from six.moves import cPickle as pickle
from six.moves import range

def weight_variables(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1],
                 strides = [1,2,2,1], padding='SAME')
    

if __name__=='__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with tf.variable_scope('Placeholder'):
        x = tf.placeholder(tf.float32, [None,784], name='inputs_placeholder')
        y_ = tf.placeholder(tf.float32, [None,10], name='labels_placeholder')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('NN'):
        W_conv1 = weight_variables('W_conv1', [5,5,1,32])
        b_conv1 = bias_variable('b_conv1', [32])

        x_image = tf.reshape(x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, W_conv1),b_conv1))
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variables('W_conv2', [5,5,32,64])
        b_conv2 = bias_variable('b_conv1', [64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variables('W_fc1', [7*7*64, 1024])
        b_fc1 = bias_variable('b_fc1', [1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variables('W_fc2', [1024,10])
        b_fc2 = bias_variable('b_fc2', [10])

        y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name='output')

    with tf.variable_scope('Loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    
    with tf.variable_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1, name='prediction'), 
                                      tf.argmax(y_,1),name='correct_predictions')

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    #define scalar_summary(tensor board)
    tf.scalar_summary('loss', cross_entropy)
    tf.scalar_summary('accuracy', accuracy)
    merged_summary_op = tf.merge_all_summaries()
    
 
    with tf.Session() as sess:            
        sess.run(init)

        #initialize summary_writer (tensor board)
        summary_writer = tf.train.SummaryWriter('log_cnn_stats', 
                                                graph=tf.get_default_graph())

        #initialize relevant variables to training
        avg_cost = 0        
        batch_size = 50
        total_batch = int(mnist.train.num_examples/batch_size)
        display_step = 100

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([train_step, cross_entropy, merged_summary_op], 
                                     feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

            summary_writer.add_summary(summary, i)

            avg_cost += c / total_batch 

            if (i+1) % display_step == 0:
                print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        #test model
        print("Accuracy:", accuracy.eval({x: mnist.test.images, 
                                     y_: mnist.test.labels, 
                                     keep_prob: 1.0}))

        print("Run the command line:\n" \
              "--> tensorboard --logdir=/tmp/tensorflow_logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser")

        #save model
        saver.save(sess, 'tf_cnn_model.ckpt')

