#!/bin/bash
echo "running PyTorch implementation..."
python ./src/pytorch/dataset_example.py
python ./src/pytorch/mnist_cnn_classifier.py
python ./src/pytorch/resnet_50.py
python ./src/pytorch/transformer.py

echo "running TensorFlow implementation..."
python ./src/tensor_flow/dataset_example.py
python ./src/tensor_flow/mnist_cnn_classifier.py
python ./src/tensor_flow/resnet_50.py
python ./src/tensor_flow/transformer.py

echo "all examples finished."
