#!/bin/bash

# 顺次执行Python脚本
echo "Data preprocessing"
python ./O2/train20000/datadivison/datadivision.py && \
python ./O2/train20000/datadivison/datadivision_sequence.py && \
python ./O2/train20000/datadivison/datadivision_test_error_len64.py && \
cd ./O2/train20000/Proposed/Scheme_trainon_45conds && \
#echo "Model Training"
#python train.py && \
echo "Testing"
python test.py && \

echo "All scripts executed successfully."