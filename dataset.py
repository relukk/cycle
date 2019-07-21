from glob import glob
import tensorflow as tf


class Dataset:

    def parse_function(self, filename):
        image_string = tf.read_file(filename)
        code = tf.decode_raw(image_string, tf.uint8)[0]
        img = tf.cond(tf.equal(code, 137),
                          lambda: tf.image.decode_png(image_string, channels=3),
                          lambda: tf.image.decode_jpeg(image_string, channels=3))
        # image_resized = tf.image.resize_images(image_decoded, size=(100, 100))
        return img

    def __init__(self, gs, lfile,mode):
        self.data = glob(lfile+'/*.'+mode)
        self.filenames = tf.constant(self.data)
        self.gs = gs #global step

        self.dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        self.dataset = self.dataset.map(self.parse_function)
        self.dataset = self.dataset.shuffle(buffer_size=10,seed=0).repeat()
        self.dataset = self.dataset.batch(1)

        self.iterator = self.dataset.make_one_shot_iterator()
        self.one_batch = self.iterator.get_next()



