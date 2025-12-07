# CSCI5448 Project

This repo contains documents and source code for the graduate project.

## Author

- Yutong Zhang, [yutong.zhang.2000@colorado.edu](mailto:yutong.zhang.2000@colorado.edu)
- Xujia Zhang, [xujia.zhang@colorado.edu](mailto:xujia.zhang@colorado.edu)

## Installation

- Install the environment from the provided conda `environment.yml`.

    ```bash
    conda env create -f environment.yml
    ```

## File Structure

- `doc/` contains `PDF` files of project proposal, outline and draft.
- `src/` contains source code of example implementations.
    - `src/pytorch` contains coding example in PyTorch.
    - `src/tensor_flow` contains coding example in TensorFlow.

## Run

- Run example scripts by:
  ```
  python ./src/pytorch/dataset_example.py
  python ./src/pytorch/mnist_cnn_classifier.py
  python ./src/pytorch/resnet_50.py
  python ./src/pytorch/transformer.py

  python ./src/tensor_flow/dataset_example.py
  python ./src/tensor_flow/mnist_cnn_classifier.py
  python ./src/tensor_flow/resnet_50.py
  python ./src/tensor_flow/transformer.py
  ```

- Or run all examples with thr provided shell script `./run_all.sh`.
