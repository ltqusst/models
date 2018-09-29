from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets import mobilenet_v1

import numpy as np
from datetime import datetime
import sys,os
import glob
import cv2


slim = tf.contrib.slim


tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('fine_tune_checkpoint',None,'where the fine tune base check point is')

tf.app.flags.DEFINE_string('test',None,'where the ckpt file to test')

tf.app.flags.DEFINE_integer('num_steps', 42000, 'maximum steps')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_integer('input_size', 224, 'input image size')

tf.app.flags.DEFINE_list('imglist', glob.glob("/dockerv0/data/voc/VOCdevkit/VOC2020/JPEGImages/*.jpg") , 'image list')


FLAGS = tf.app.flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94



def get_rotation_scoremap(r, field_radius = 40.0, length=1000):
    zero_pos = round(length/2)
    target = r + zero_pos

    # each node output a score propotional to it's distance to the target
    sm = np.zeros(shape=(length,), dtype=np.float32)
    for i in range(length):
        dist = float(abs(target - i))
        sm[i] = 1 - min(1.0, dist/field_radius)

    return sm


class DataSet(object):
    def __init__(self,img_file_list, input_size, batch_size):
        self.batch_size = batch_size
        self.input_size = input_size
        self.img_file_list = img_file_list
        self.sample_returned = 0

    def get_batch(self):
        # feed train data as numpy array

        # each image needs a crop resize to 224x224 for trainning
        # and this is already done in sample generation step
        # also the rotation degree is already encoded in filename
        imgs = np.random.choice(self.img_file_list, self.batch_size)

        batch_inputs = []
        batch_labels = []
        for imgpath in imgs:
            img = cv2.imread(imgpath)
            assert(img.shape == (self.input_size, self.input_size, 3))
            rot = int(imgpath.split('r')[-1].split('.')[0])
            label = get_rotation_scoremap(rot)
            batch_inputs.append(img)
            batch_labels.append(label)

        batch_inputs = np.array(batch_inputs)
        batch_labels = np.array(batch_labels)

        self.sample_returned += self.batch_size

        return batch_inputs, batch_labels

    def epoch(self):
        return self.sample_returned / len(self.img_file_list)

class Model(object):
    def __init__(self, checkpoint_dir, input_size, batch_size):
        """Builds graph for model to train with rewrites for quantization.

        Returns:
        g: Graph with fake quantization ops and batch norm folding suitable for
        training quantized weights.
        train_tensor: Train op for execution during training.
        """
        self.global_step = tf.train.get_or_create_global_step()
        self.checkpoint_dir = checkpoint_dir
        self.input_size = input_size

        g = tf.Graph()
        with g.as_default():
            # image
            inputs = tf.placeholder(tf.float32, [batch_size, self.input_size, self.input_size, 3], name="inputs")
            scoremap = tf.sparse_placeholder(tf.float32, [batch_size, 1000], name="labels")

            # make it between (-1 to 1) to match the pretrained model
            inputs1 = tf.subtract(inputs, 0.5)
            inputs1 = tf.multiply(inputs1, 2.0)

            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
                logits, _ = mobilenet_v1.mobilenet_v1(
                                  inputs1,
                                  is_training=True,
                                  depth_multiplier=1.0,
                                  num_classes=1000,
                                  prediction_fn=tf.nn.relu)

            total_loss = tf.losses.absolute_difference(scoremap, logits)

            # Call rewriter to produce graph with fake quant ops and folded batch norms
            # quant_delay delays start of quantization till quant_delay steps, allowing
            # for better model accuracy.
            #if FLAGS.quantize:
            #  tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())
            # total_loss = tf.losses.get_total_loss(name='total_loss')

            # Configure the learning rate using an exponential decay.
            num_epochs_per_decay = 2.5
            imagenet_size = 1271167
            decay_steps = int(imagenet_size / batch_size * num_epochs_per_decay)

            learning_rate = tf.train.exponential_decay(
                1e-4,
                self.global_step,
                decay_steps,
                _LEARNING_RATE_DECAY_FACTOR,
                staircase=True)
            opt = tf.train.GradientDescentOptimizer(learning_rate)

            train_tensor = slim.learning.create_train_op(
                total_loss,
                optimizer=opt)

        self.total_loss = total_loss
        self.g = g
        self.train_op = train_tensor
        self.inputs = inputs
        self.labels = scoremap
        self.name = "rotReg"

        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('learning_rate', learning_rate)

        self.merged_summay = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


    def go1step(self, sess, feed, train_writer = None):

        # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
        assert (self.inputs in feed)
        assert (self.labels in feed)

        summary_str, batch_cost, step, _ = \
            sess.run([self.merged_summay, self.cost, self.global_step, self.train_op], feed)

        print("step {}: batch_cost={}".format(step, batch_cost) + " " * 10, end='\r')
        sys.stdout.flush()

        if train_writer:
            train_writer.add_summary(summary_str, step)

        if step % 1000 == 1:
            if not os.path.isdir(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            print('save checkpoint at step {0}'.format(step))
            self.saver.save(sess, os.path.join(self.checkpoint_dir, self.name), global_step=step)

        '''    
        if step % FLAGS.validation_steps == 0:
            # random test one sample
            batch_inputs, batch_labels, batch_seqlen = dataset.get_batch(indexs[0:1])
            feed = {model.inputs: batch_inputs,
                    model.labels: batch_labels}
            dense_decoded0, dense_decoded1 = sess.run([model.dense_decoded0, model.dense_decoded1], feed)

            print("sample {:6}:{}".format(i, dataset.seq2text(dataset.get_seq(indexs[0]))))
            print("decoded0     :{}".format(dataset.seq2text(dense_decoded0[0])))
            print("decoded1     :{}".format(dataset.seq2text(dense_decoded1[0])))
        '''


def Train(sess, model, ds):
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, '::1:6034')
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    step = sess.run([model.global_step])[0]
    epoch = 0

    print('>>>>>>>>>>>>>>>>>> begin training from step {} up to maximum steps {}'.format(step, FLAGS.num_steps))
    now = datetime.now()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + model.name + now.strftime("(%m%d-%H%M%S)"),
                                         sess.graph)

    while(step < FLAGS.num_steps):
        epoch += 1
        shuffle_idx = np.random.permutation(num_train_samples)
        train_cost = 0

        print('epoch:{} '.format(ds.epoch()))

        batch_inputs, batch_labels = ds.get_batch()

        feed = {model.inputs: batch_inputs,
                model.labels: batch_labels}

        model.go1step(sess, feed, train_writer)


def main(unused_arg):
    """Trains mobilenet_v1."""

    model = Model(FLAGS.checkpoint_dir, FLAGS.input_size, FLAGS.batch_size)
    ds = DataSet(FLAGS.imglist, FLAGS.input_size, FLAGS.batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        if FLAGS.test:
            ckpt = tf.train.latest_checkpoint(FLAGS.test)
            assert ckpt is not None

            saver.restore(sess, ckpt) # the global_step will restore sa well
            print('>>>>>>>>>>>>>>>>>> restored from checkpoint{0}'.format(ckpt))

            #Test(sess, dataSet, model, FLAGS.checkpoint_dir)
        else:
            Train(sess, model, ds)


if __name__ == '__main__':
  tf.app.run(main)
