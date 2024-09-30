# Robust Heterogeneous Graph Neural Network (RoHeHAN)

This is an implementation of `RoHeHAN`, a robust heterogeneous graph neural network designed to defend against adversarial attacks on heterogeneous graphs.

- Paper link: [https://cdn.aaai.org/ojs/20357/20357-13-24370-1-2-20220628.pdf](https://cdn.aaai.org/ojs/20357/20357-13-24370-1-2-20220628.pdf)
- Original paper title: *Robust Heterogeneous Graph Neural Networks against Adversarial Attacks*
- Implemented using `tensorlayerx` and `gammagl` libraries.

## Usage

To reproduce the RoHeHAN results on the ACM dataset, run the following command:

```bash
TL_BACKEND="torch" python rohehan_trainer.py --num_epochs 100 --gpu 0
```

## Performance

Reference performance numbers for the ACM dataset:

| Dataset | Clean (no attack) | Attack(1 perturbation) | Attack(3 perturbations) | Attack(5 perturbations) |
| ------- | ----------------- | ---------------------- | ----------------------- | ----------------------- |
| ACM     | 0.930             | 0.915                  | 0.905                   | 0.895                   |

ACM dataset link: [https://github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat](https://github.com/Jhy1993/HAN/raw/master/data/acm/ACM.mat)

### Example Commands

You can adjust training settings, such as the number of epochs, learning rate, and dropout rate, with the following commands:

```bash

TL_BACKEND="torch" python rohehan_trainer.py --num_epochs 200 --lr 0.005 --dropout 0.6 --gpu 0

```

## Notes

- Early stopping is used to prevent overfitting during training.
- The `settings` in the RoHeGAT layer control the attention purifier mechanism, which ensures robustness against adversarial attacks by pruning unreliable neighbors.

This implementation builds on the idea of using metapath-based transiting probability and attention purification to improve the robustness of heterogeneous graph neural networks (HGNNs).