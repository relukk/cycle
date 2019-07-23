import tensorflow as tf
from model import *
from argparse import ArgumentParser


def main(_):
    args = ArgumentParser()
    args.add_argument(
        '-p',
        '--fname',
        required=True,
        help='path to dataset'
    )
    args.add_argument(
        '-c',
        '--image_code',
        required=True
    )

    args = args.parse_args()

    model = Cyclegan(args)
    with tf.Session() as sess:
        model.train(sess)


if __name__ == '__main__':
    print('1')
    tf.app.run()
