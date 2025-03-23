# DeepPoly


This project implements the DeepPoly algorithm introduced in "An Abstract Domain for Certifying Neural Networks". It uses [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation) to check if neural networks trained on MNIST and CIFAR10 are robust against input perturbations. The analyzer supports feedforward, convolutional, and ReLU layers, automatically verifying whether specified safety properties hold under different kinds of changes to the input.
This work was carried out for the course “Reliable and Trustworthy Artificial Intelligence” at ETH Zurich.

<p align="center">
    <img src="https://imgur.com/ejEE2wE.png" alt="overview" width="100%">
</p>




## Setup

Create a virtual environment and install the dependencies using:

```
conda env create -f ./environment.yaml
conda activate rtai-project
```

If you prefer pip:
```
virtualenv venv --python=python3.10
source venv/bin/activate
pip install -r requirements.txt
```

## How to run

You can run the verifier with:

```bash
python code/verifier.py \
    --net {net} \
    --spec test_cases/{net}/img{id}_{dataset}_{eps}.txt \
    --labels path/to/labels.txt
```

where, 
- `net` can be one of: `fc_base`, `fc_1`, `fc_2`, `fc_3`, `fc_4`, `fc_5`, `fc_6`, `fc_7`, `conv_base`, `conv_1`, `conv_2`, `conv_3`, `conv_4`.
- `spec` is the path to the chosen test case, which includes:
    - `id`: a numerical identifier for the case.,
    - `dataset`: the dataset name, i.e.,  either `mnist` or `cifar10`,
    - `eps`: the perturbation level to be certified.
- `labels` (optional) is the path to a file containing information on whether each test case is verifiable or not (used for evaluation).

For example:

```bash
python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt
```

To evaluate the verifier on all networks and sample test cases, run:
```bash
scripts/evaluate
```