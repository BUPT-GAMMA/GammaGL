# Calibrating Graph Neural Networks from a Data-centric Perspective (DCGC)

This example implements the method from the paper "Calibrating Graph Neural Networks from a Data-centric Perspective".

## Running the Example

```bash
# Run DCGC on the Pubmed dataset
python dcgc_main_result.py --dataset Pubmed

# Run DCGC on the Cora dataset
python dcgc_main_result.py --dataset Cora

# Run DCGC on the CiteSeer dataset
python dcgc_main_result.py --dataset CiteSeer
```

## Parameter Description

- `--dataset`: Dataset used for training, default is "Pubmed"
- `--epochs`: Number of epochs to train the base model, default is 1000
- `--cal_epochs`: Number of epochs to train the calibration model, default is 1000
- `--lr`: Initial learning rate, default is 0.01
- `--lr_for_cal`: Learning rate for the calibration model, default is 0.01
- `--hidden`: Number of hidden units, default is 16
- `--dropout`: Dropout rate, default is 0.7
- `--weight_decay`: Weight decay (L2 regularization), default is 5e-4
- `--l2_for_cal`: Weight decay for the calibration model, default is 5e-3
- `--n_bins`: Number of bins for ECE calculation, default is 20
- `--patience`: Patience for early stopping, default is 100
- `--num1`: Number of times to train the base model, default is 1
- `--num2`: Number of times to train the calibration model on each base model, default is 5
- `--alpha`: Alpha parameter in edge weight calculation, default is 0.5
- `--beta`: Beta parameter in edge weight calculation, default is 10
- `--gpu`: GPU ID to use, default is 0

## Implemented Calibration Methods

1. TS: Temperature Scaling
2. TS+EW: Temperature Scaling + Edge Weight

