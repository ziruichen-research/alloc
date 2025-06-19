#!/bin/bash

# 顺次执行Python脚本
echo "Data preprocessing"
python ./RO1/train40000/datadivison/datadivision.py && \
python ./RO1/train40000/datadivison/datadivision_sequence.py && \
python ./RO1/train40000/datadivison/datadivision_test_error_len64.py && \
cd ./RO1/train40000/Proposed/Scheme && \
#echo "Model Training"
#python train.py && \
echo "Testing"
python test.py && \

echo "All scripts executed successfully."