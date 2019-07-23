import tensorflow as tf
import numpy as np
import time
import os
from util import *
import config
from save import *
from dataset import *
from plotimage import *
from dataset import *


class Cyclegan:
    def __init__(self, args):
        self.bsz = config.bsz
        self.height = config.height
        self.width = config.width
        self.fl = config.fo
        self.lr = config.lr
        self.gs = tf.Variable(initial_value=0,name='global_step')
        self.channel = config.channel
        self.beta = config.beta
        self.epoch = config.epoch
        self.lamda = config.lamda
        self.data = None
        self.ndata = 0 # of training images
        self.batch_img = tf.zeros([self.bsz,self.height,self.width,self.channel],dtype=tf.float32)
        self.lfile = args.fname
        self.img_code = args.image_code
        self.img_dir = 'out/'
        self.log_dir = 'log/'
        self.ck_dir = 'checkpoint/'
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.ck_dir):
            os.mkdir(self.ck_dir)
        self.build_model()

    def generator(self, inp, name, reuse):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            paddings = tf.constant([[0,0],[3,3],[3,3],[0,0]])
            pad = tf.pad(inp,paddings=paddings,mode='REFLECT')
            w = tf.get_variable(name='l1_w',shape=[7,7,self.channel,self.fl])
            l1 = tf.nn.conv2d(pad,w,[1,1,1,1],padding='VALID',name='l1')
            l2 = dk(l1, self.fl*2, name='l2') #128
            l3 = dk(l2, self.fl*4, name='l3') #256
            lin = l3
            for i in range(0,9):
                l = rk(lin, self.fl*4, name='l'+str(i+4))
                lin = l
            l12 = l
            l13 = uk(l12,self.fl*2,self.fl*2,name='l13')
            l14 = uk(l13,self.fl*4,self.fl,name='l14')
            w = tf.get_variable(name='out_w',shape=[7,7,l14.get_shape()[-1],self.channel])
            out = tf.nn.conv2d(l14,w,[1,1,1,1],padding='SAME',name='out')
        return out

    def discriminator(self,inp,name,reuse):
        with tf.variable_scope(name_or_scope=name,reuse=reuse):
            l = lrelu(conv2d(inp,self.fl,name='l1',k_h=4,k_w=4))
            for i in range(2,5):
                l = ck(l, self.fl*(2**(i-1)),name='l'+str(i))
            out = conv2d(l,1,name='out',k_h=4,k_w=4)
        return out

    def build_model(self):
        self.data = tf.placeholder(dtype=tf.float32,shape=[self.bsz, self.height, self.width, self.channel],name='data')
        self.photo = self.data[:,:,0:self.height,:]
        self.label = self.data[:,:,self.height:,:]
        self.fake_x = self.generator(self.label,name='g_g',reuse=False)#(G(x)) should generate photo
        self.fake_y = self.generator(self.fake_x,name='g_f',reuse=False)#(F(G(x))) should generate label
        self.r_x = self.generator(self.photo,name='g_f',reuse=True)#F(y)  should generate label
        self.r_y = self.generator(self.r_x,name='g_g',reuse=True)#G(F(y)) should generate photo
        self.d = self.discriminator(self.photo,name='d_y',reuse=False)#D(y)
        self.dd = self.discriminator(self.label,name='d_x',reuse=False)#D(x)
        self.d_x = self.discriminator(self.fake_y,name='d_x',reuse=True)#D(F(G(x)))
        self.d_y = self.discriminator(self.fake_x,name='d_y',reuse=True)
        self.loss = GANLoss('lsgan')

        self.lsgan_gg = self.loss(self.d_y,1) #positive cycle
        self.lsgan_gf = self.loss(self.d_x,tf.ones_like(self.d_x))
        self.lsgan_dy = 0.5*(self.loss(self.d,tf.ones_like(self.d))+self.loss(self.d_y,tf.zeros_like(self.d_y)))
        self.lsgan_dx = 0.5*(self.loss(self.dd,tf.ones_like(self.dd))+self.loss(self.d_x,tf.zeros_like(self.d_x)))
        self.cl = CycleLoss(self.fake_y,self.label)
        self.ccl = self.cl(self.fake_y,self.label)+self.cl(self.r_y,self.photo)

        self.gg_loss = self.lsgan_gg+self.lamda*self.ccl
        self.gf_loss = self.lsgan_gf+self.lamda*self.ccl
        self.dy_loss = self.lsgan_dy
        self.dx_loss = self.lsgan_dx

        vars = tf.trainable_variables()
        self.dy_vars = [var for var in vars if 'd_y' in var.name]
        self.gg_vars = [var for var in vars if 'g_g' in var.name]
        self.dx_vars = [var for var in vars if 'd_x' in var.name]
        self.gf_vars = [var for var in vars if 'g_f' in var.name]

        tf.summary.scalar('gg_loss', self.gg_loss)
        tf.summary.scalar('gf_loss', self.gf_loss)
        tf.summary.scalar('dy_loss', self.dy_loss)
        tf.summary.scalar('dx_loss', self.dx_loss)

        self.saver = tf.train.Saver()

    def train(self,sess):
        self.sess = sess
        self.d_y = tf.train.AdamOptimizer(self.lr, beta1=self.beta).minimize(self.dy_loss, var_list=self.dy_vars)
        self.g_g = tf.train.AdamOptimizer(self.lr, beta1=self.beta).minimize(self.gg_loss, var_list=self.gg_vars)
        self.d_x = tf.train.AdamOptimizer(self.lr, beta1=self.beta).minimize(self.dx_loss, var_list=self.dx_vars)
        self.g_f = tf.train.AdamOptimizer(self.lr, beta1=self.beta).minimize(self.gf_loss, var_list=self.gf_vars)

        data = glob(self.lfile+'/*.'+self.img_code)
        self.ndata = len(data)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.ck_dir)
        print("ckpt", ckpt)
        self.saver.export_meta_graph(os.path.join(self.ck_dir, "cyclegan.model.meta"))

        if ckpt and ckpt.model_checkpoint_path:
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            gs = self.sess.run(self.gs)
            print("restore")
        else:
            self.sess.run(tf.global_variables_initializer())
            gs = 0

        self.dataset = Dataset(gs,self.lfile,self.img_code)
        start_time = time.time()
        totalbatch = self.ndata // self.bsz
        e = gs // totalbatch

        while e < self.epoch and not stop:
            if e < 100:
                self.lr = config.lr
            else:
                self.lr = config.lr - config.lr * (e-100)/100
            index = gs % totalbatch + 1
            while index <= totalbatch and not stop:
                try:
                    self.batch_img = sess.run(self.dataset.one_batch)
                except tf.errors.OutOfRangeError:
                    print('End')
                self.sess.run([self.g_g,self.d_x,self.g_f,self.d_y], feed_dict={self.data: self.batch_img})
                gg_loss, gf_loss, dy_loss, dx_loss = self.sess.run([self.gg_loss,self.gf_loss,self.dy_loss,self.dx_loss],feed_dict={self.data:self.batch_img})
                gs += 1
                self.sess.run(self.gs.assign(gs))
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, gg_loss: %.8f, gf_loss: %.8f, dy_loss: %.8f, dx_loss: %.8f" \
                      % (e, self.epoch, index, totalbatch, time.time() - start_time, gg_loss,gf_loss,dy_loss,dx_loss))
                if gs % 100 == 1:
                    summary = self.sess.run(merged, feed_dict={self.data: self.batch_img})
                    train_writer.add_summary(summary, gs)
                    try:
                        self.sample_img = self.sess.run(self.dataset.one_batch)
                    except tf.errors.OutOfRangeError:
                        print("End")
                    samples = self.sess.run(self.fake_x, feed_dict={self.data: self.sample_img})
                    save_images(samples, image_manifold_size(samples.shape[0]), \
                                './{}/train_{:02d}_{:04d}.png'.format('out/', e, index))
                    # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (loss_d_real+loss_d_fake, loss_g))
                    print('save image')

                if gs % 500 == 2:
                    save(checkpoint_dir=self.ck_dir, saver=self.saver, sesss=self.sess, step=self.gs)
                index += 1
            e = e + 1
        save(checkpoint_dir=self.ck_dir, saver=self.saver, sesss=self.sess, step=self.gs)
        train_writer.close()














