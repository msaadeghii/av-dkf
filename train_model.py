#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

Adapted and modified by Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""

import sys
from dvae import LearningAlgorithm
from test_speech_analysis_resynthesis import evaluate

if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        learning_algo = LearningAlgorithm(config_file=cfg_file)
        path = learning_algo.train()
        
        # Evaluate the model
        evaluate(path, 10)

    else:
        print('Error: Please input the config file')
