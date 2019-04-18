#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import os
import data_helpers
import time
import datetime
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "rt-polaritydata/rt-polarity.neg", "Data source for the negative data.") # Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)") # Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)") # Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
learning_rate=1e-3
class TextCNN(object):
    def __init__(self,sequence_len,num_class,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):
        #sequence_len句子长度
        self.input_x=tf.placeholder(tf.int32,[None,sequence_len])
        self.input_y=tf.placeholder(tf.float32,[None,num_class])
        self.dropout_keep_prob=tf.placeholder(tf.float32)

        l2_loss=tf.constant(0.0)

        #with tf.variable_scope('embedding'):
        self.W=tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
        self.embeded=tf.nn.embedding_lookup(self.W,self.input_x)
        self.embeded_expended=tf.expand_dims(self.embeded,-1)   #增加一个表示通道数的维度，把2维的语言数据，扩展成和图片一样的数据

        pooled_output=[]
        for i,filter in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s'% filter):
                filter_shape=[filter,embedding_size,1,num_filters]
                W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
                b=tf.Variable(tf.constant(0.1,shape=[num_filters]))
                #print b.get_shape().as_list(),'shaaaaaa'
                conv=tf.nn.conv2d(
                    self.embeded_expended,
                    W,
                    strides=[1,1,1,1],
                    padding='VALID'

                )
                #print conv.get_shape().as_list(), 'zxcccccc'
                #输出是128（num_filters）个一纵条一
                #self.embeded_expended必须是一个4维的向量
                #W也是一个4维向量，分别代表长宽，第三维是图片的通道数，黑白是1，RGB是3,语言的话是1,第4维是输出的通道数，就是你输出的长方体的厚度
                h=tf.nn.relu(tf.nn.bias_add(conv,b))
                #bias_add是一个神奇的函数，bias即b必须是一维的，这个函数支持广播的形式，所以value即conv，可以是任何形状维度
                pooled=tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_len-filter+1,1,1],
                    strides=[1,1,1,1],
                    padding='VALID'

                )
                #h是需要池化的输入,  形状是[batch, height, width, channels]这样的shape,结构是一纵条，这个池化即是选出卷积层中每一个卷积核最大的一个值，最后拼接
                #ksizem池化窗口的大小,[1, height, width, 1]，因为我们不想在batch和channels上做池化
                #strides,和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                print pooled
                pooled_output.append(pooled)
            #第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature

            #第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

            #第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]

            #第四个参数padding：和卷积类似，可以取'VALID'或者'SAME'

            #返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]
            #这种形式

        num_filters_total = num_filters * len(filter_sizes)  # 例如3,4,5的卷积核，每个卷积核有2个一共有6个，128×3
        print pooled_output,'iiiqqq'
        self.h_pool=tf.concat(pooled_output,3)          #3代表一个维度,即最后一个维度，相当于所有池化结果拼成一列.!!!!tensor列表可以拼接
        print self.h_pool.get_shape().as_list(),'bbbbtttt'

        self.h_pool_flat=tf.reshape(self.h_pool,[-1,num_filters_total])
        print self.h_pool_flat.get_shape().as_list(),'vvccccccc'

        with tf.variable_scope('dropout'):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)  #dropout
        #with tf.variable_scope('output'):
            # W = tf.get_variable("W", shape=[num_filters_total, num_class],
            #                     initializer=tf.contrib.layers.xavier_initializer())#这句化的作用

        W=tf.Variable(tf.random_uniform([num_filters_total, num_class],-1.0,1.0))#全连接层的W系数
        #num_class这里是2,2分类

            #get_variable可用于变量共享：
            #例如：
            # with tf.variable_scope("scope1"):
            #     w1 = tf.get_variable("w1", shape=[])
            #     w2 = tf.Variable(0.0, name="w2")
            # with tf.variable_scope("scope1", reuse=True):
            #     w1_p = tf.get_variable("w1", shape=[])
            #     w2_p = tf.Variable(1.0, name="w2")
            #
            # print(w1 is w1_p, w2 is w2_p)
            # 输出
            # True  False






        b=tf.Variable(tf.constant(0.1,shape=[num_class]))
        l2_loss+=tf.nn.l2_loss(W)           #这个函数的作用
        #tf.nn.l2_loss(t, name=None)
        #上述函数的计算过程是这样的：output = sum(t ** 2) / 2
        l2_loss+=tf.nn.l2_loss(b)
        #以上两步是正则化的准备工作
        self.scores=tf.matmul(self.h_drop,W)+b #h乘w
        #全连接层，N*两列。前面哪个矩阵的行，后面那个矩正的列
        print self.scores.get_shape().as_list(),'ffgggggg'
        self.predictions=tf.argmax(self.scores,1)   #1是说明维度即self.scores里最大的下标，就是类别，每一行里最大的那一列，返回下标号。
        #没法预测值的范围，只使得正确的值拥有较大的预测值
            # test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
            # np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
            # np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]



        #with tf.variable_scope("loss"):
        losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)#注意交叉熵要直接用预测出来的结果计算
        self.loss=tf.reduce_mean(losses)+l2_reg_lambda*l2_loss  #加了正则化项的损失函数

        #with tf.variable_scope('accurary'):
        correct_predict=tf.equal(self.predictions,tf.argmax(self.input_y,1))#相等是1,不等是0。predictions是预测类别
        self.accuracy=tf.reduce_mean(tf.cast(correct_predict,'float'))#tf.cast是强制类型转换，计算准确率

        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)   #选择优化函数，确定学习率
        #self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        #对于在变量列表（var_list）中的变量计算对于损失函数的梯度,这个函数返回一个（梯度，变量）对的列表，其中梯度就是相对应变量的梯度了。这是minimize()函数的第一个部分，
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
        ##apply_gradients的参数是(gradient, variable)对的列表；(gradient, variable)是compute_gradients()的结果
        #以上两步相当于optimizer。minimize（self.loss）



def train(model,sess,x_train, y_train, x_dev, y_dev):
    #TensorFlow中最重要的可视化方法是通过tensorBoard、tf.summary和tf.summary.FileWriter这三个模块相互合作来完成的
    #tf.summary模块的定义位于summary.py文件中，该文件中主要定义了在进行可视化将要用到的各种函数，
    #grad_summaries=[]
    for g, v in model.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #入的Tensor中0元素在所有元素中所占的比例计算并返回，因为relu激活函数有时会大面积的将输入参数设为0，所以此函数可以有效衡量relu激活函数的有效性。
            # grad_summaries.append(grad_hist_summary)
            # grad_summaries.append(sparsity_summary)

    loss_summary = tf.summary.scalar("loss", model.loss)#tf.summary.scalar()函数的功能理解为：
    #[1]将【计算图】中的【标量数据】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备

    acc_summary = tf.summary.scalar("accuracy", model.accuracy)

    # Train Summaries # 训练数据保存
    #Tensorflow提供了tf.summary.merge_all()函数将所有的summary整理在一
    #起。在TensorFlow程序执行的时候，只需要运行这一个操作就可以将代码中定义的所有【写日志操作】执行一次，从而将
    #所有的日志写入【日志文件】。

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    #将Summary protocol buffers写入磁盘文件
    #FileWriter类提供了一种用于在给定目录下创建事件文件的机制


    summaryOp = tf.summary.merge_all()
    #  Dev summaries # 测试数据保存
    #dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)



    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=FLAGS.num_checkpoints)  # Write vocabulary vocab_processor.save(os.path.join(out_dir, "vocab"))


    batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(model,sess,x_batch, y_batch,train_summary_writer,summaryOp)
        current_step = tf.train.global_step(sess, model.global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(model,sess,x_dev, y_dev, summaryOp,writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix,global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

def train_step(model,sess,x_batch, y_batch,train_summary_writer,train_summary_op):
    feed_dict={
        model.input_x:x_batch,
        model.input_y:y_batch,
        model.dropout_keep_prob:FLAGS.dropout_keep_prob
    }
    _,step,summaries,loss,accuracy=sess.run(
        [model.train_op,model.global_step,train_summary_op,model.loss,model.accuracy],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()  # 取当前时间，python的函数
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)

def dev_step(model,sess,x_batch, y_batch, dev_summary_op,writer=None):
    feed_dict = { model.input_x: x_batch, model.input_y: y_batch, model.dropout_keep_prob: 1.0 }
    step, summaries, loss, accuracy = sess.run(
        [model.global_step,  dev_summary_op,model.loss, model.accuracy],
        feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    if writer: writer.add_summary(summaries, step)

def main(_):
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file,FLAGS.negative_data_file)  # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text]) # 计算最长

    #这步已经实现了单词到对应编码的转换
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length) # tensorflow提供的工具，将数据填充为最大长度，默认0填充
    x = np.array(list(vocab_processor.fit_transform(x_text))) # Randomly shuffle data # 数据洗牌
    # x_text = [
    #     'i love you',
    #     'me too'
    # ]
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # vocab_processor.fit(x_text)
    # print next(vocab_processor.transform(['i me too'])).tolist()
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # print x
    #
    # [1, 4, 5, 0]
    # [[1 2 3 0]
    #  [4 5 0 0]]

    np.random.seed(10) # np.arange生成随机序列
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices] # 将数据按训练train和测试dev分块
    #  Split train/test set
    #  This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev))) # 打印切分的比例
    print y_train.shape[1],'labellabel'
    cnn=TextCNN(sequence_len=x_train.shape[1],num_class=y_train.shape[1],vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
                 )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train(cnn,sess,x_train,y_train,x_dev,y_dev)
if __name__=='__main__':
    tf.app.run()
