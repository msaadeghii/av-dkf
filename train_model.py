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
vf_root = "/srv/storage/talc2@talc-data2.nancy/multispeech/calcul/users/agolmakani/.cache/vf_all_OLD"
if __name__ == '__main__':

    if len(sys.argv) == 2:
        cfg_file = sys.argv[1]
        learning_algo = LearningAlgorithm(config_file=cfg_file)
        path = learning_algo.train(vf_root)

        print(path)
        evaluate(path, 1000)

    else:
        print('Error: Please indiquate config file')
