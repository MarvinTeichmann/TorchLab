"""
The MIT License (MIT)

Copyright (c) 2021 Marvin Teichmann
Email: marvin.teichmann@googlemail.com

The above author notice shall be included in all copies or
substantial portions of the Software.

This file is written in Python 3.8 and tested under Linux.
"""

import logging
import sys
import torch


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


def normalize_training_config(conf, check_gpus=True):
    """
    Normalize training hyperparameters with respect to gpus and batch_size.

    We are applying the following rules:
    batch_size ~ num_gpus
    num_workers ~ num_gpus
    learning_rate ~ batch_size

    The rules are descripted in the paper:
    [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](
        https://arxiv.org/abs/1706.02677v2)
    """

    num_gpus = conf['training']['num_gpus']

    if check_gpus:
        assert num_gpus == torch.cuda.device_count()

    if num_gpus == 0:
        # Training happens on CPU.
        # Assuming single CPU training.
        num_gpus = 1

    conf['dataset']['num_workers'] *= num_gpus
    conf['training']['batch_size'] *= num_gpus

    conf['training']['learning_rate'] *= conf['training']['batch_size']
    conf['training']['min_lr'] *= conf['training']['batch_size']




if __name__ == '__main__':
    logging.info("Hello World.")
