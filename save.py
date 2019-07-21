import os


def save(checkpoint_dir, step, saver, sesss):
    model_name = 'cycleGAN.model'
    saver.save(sesss, os.path.join(checkpoint_dir, model_name), global_step=step)
