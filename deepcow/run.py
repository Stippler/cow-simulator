#!/usr/bin/python3

from deepcow.loops import *
from deepcow.user_play import user_play
import tensorflow as tf
import sys


def main(argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

    if len(argv) == 0:
        play_game()
    elif len(argv) != 1:
        print("usage: run.py {play, train_cow, train_wolf, train_both}")
    else:
        arg = argv[0]
        if arg == "play":
            play_game()
        elif arg == "train_cow":
            train_cow()
        elif arg == "train_wolf":
            train_wolf()
        elif arg == "train_both":
            train_both()
        elif arg == "user_play":
            user_play()
        elif arg == "plot_model":
            model = ExtendedDQNAgent(20, 3, 4, None)
            model.plot_model('QNetwork.png')


if __name__ == "__main__":
    main(sys.argv[1:])
