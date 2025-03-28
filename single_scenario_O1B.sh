#!/bin/bash

# 顺次执行Python脚本
echo "Data preprocessing"
python ./O1B/train40000/datadivision/datadivision.py && \
python ./O1B/train40000/datadivision/datadivision_sequence.py && \
python ./O1B/train40000/datadivision/datadivision_test_error_len64.py && \
cd ./O1B/train40000/Proposed/Scheme && \
#echo "Model Training"
#python train.py && \
echo "Testing"
python test.py

echo "All scripts executed successfully."