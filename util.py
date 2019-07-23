import tensorflow as tf
import signal

stop = False
_orig = None


def handle(_, __):
    global stop
    stop = True
    signal.signal(signal.SIGINT, _orig)


_orig = signal.signal(signal.SIGINT, handle)


def conv2d(input, outdim, name, pad='SAME', k_h=3,k_w=3,s_h=2,s_w=2):
    with tf.variable_scope(name):
        w = tf.get_variable('w',shape=[k_h,k_w,input.get_shape()[-1], outdim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.conv2d(input,w,[1,s_h,s_w,1],padding=pad)
    return out


def deconv2d(input, outshape, name, pad, k_h=3,k_w=3,s_h=2,s_w=2):
    with tf.variable_scope(name):
        w = tf.get_variable('w',shape=[k_h,k_w,outshape[-1],input.get_shape()[-1]],initializer=tf.truncated_normal_initializer(stddev=0.02))
        out = tf.nn.conv2d_transpose(input,w,outshape,[1,s_h,s_w,1],pad)
    return out


# convolution-instanceNorm-ReLU
def dk(input, outdim, name, k_h=3, k_w=3, s_h=2,s_w=2):
    with tf.variable_scope(name):
        # paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
        # pad = tf.pad(input, paddings=paddings, mode='REFLECT')
        l1 = conv2d(input, outdim, name+'/dk', 'SAME', k_h, k_w, s_h, s_w)
        mean, var = tf.nn.moments(l1,axes=[0,1,2])
        l1 = tf.nn.batch_normalization(l1,mean,var,None,None, 1e-4)
        out = tf.nn.relu(l1)
    return out


# fractional-strided Convolution - InstanceNorm-ReLU
def uk(input, outsize, outdim, name, k_h=3, k_w=3, s_h=2,s_w=2):
    with tf.variable_scope(name):
        # paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
        # pad1 = tf.pad(input,padding=paddings,mode='REFLECT')
        l1 = deconv2d(input, [input.get_shape().as_list()[0], outsize, outsize, outdim],name+'/uk','SAME',k_h, k_w, s_h, s_w)
        mean, var = tf.nn.moments(l1,axes=[0,1,2])
        l1 = tf.nn.batch_normalization(l1, mean, var, None, None, 1e-4)
        out = tf.nn.leaky_relu(l1)
    return out


#residual block
def rk(input, outdim, name):
    with tf.variable_scope(name):
        paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
        pad1 = tf.pad(input,paddings=paddings,mode='REFLECT')
        l1 = conv2d(pad1,outdim,name=name+'/rk/l1',pad='VALID',s_h=1,s_w=1)
        pad2 = tf.pad(l1,paddings=paddings,mode='REFLECT')
        l2 = conv2d(pad2,outdim,name=name+'/rk/l2',pad='VALID',s_h=1,s_w=1)
    return lrelu(l2+input)


#ck
def ck(input, outdim, name):
    with tf.variable_scope(name):
        l = conv2d(input,outdim,name+'/cl',k_h=4,k_w=4)
        mean, var = tf.nn.moments(l,axes=[0,1,2])
        l = tf.nn.batch_normalization(l,mean,var,None,None,1e-4)
        out = lrelu(l)
    return out


def lrelu(inp,alpha=0.2):
    return tf.math.maximum(inp, alpha*inp)


def l1_loss(label,prediction):
    return tf.losses.absolute_difference(label,prediction)


class GANLoss:
    def __init__(self, model):
        self.model = model
        if model == 'vanilla':
            print('vanilla')
        elif model == 'lsgan':
            print('lsgan')
        else:
            self.loss = None

    def __call__(self, output, label):
        if self.model == 'lsgan':
            loss = tf.reduce_mean(tf.squared_difference(output, label))
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(output, label)

        return loss


class CycleLoss:
    def __init__(self,logit,label):
        self.cycle_logit = logit
        self.cycle_label = label

    def __call__(self, logits, label):
        return tf.reduce_mean(l1_loss(logits,label) + l1_loss(logits,label))

