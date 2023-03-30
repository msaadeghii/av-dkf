#!/bin/bash

# Run speech enhancement on each segment of test files
oarsub -S "./SE_eval.sh 0"
oarsub -S "./SE_eval.sh 1"
oarsub -S "./SE_eval.sh 2"
