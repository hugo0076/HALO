# HALO: Robust Out-of-Distribution Detection via Joint Optimisation

This repository contains the code for the paper "HALO: Robust Out-of-Distribution Detection via Joint Optimisation".

## Requirements
- `Python` 3.11.6
- `pytorch` 2.0.1
- `pytorch-cuda` 12.1 
- `torchvision` 0.16.1
- `numpy` 1.26.4
- `tqdm` 4.62.3
- `openood` 1.5


## Data
For training, we use the following datasets:
- CIFAR-10 (comes with torchvision)
- CIFAR-10S (extra data): The 500k extra image can be downloaded from: https://github.com/yaircarmon/semisup-adv. Place file at `data/ti_500k_pseudo_labeled.pickle`.
- CIFAR-100 (comes with torchvision)
- Tiny-ImageNet: downloadable from: http://cs231n.stanford.edu/tiny-imagenet-200.zip. Download and extract to `data/tiny-imagenet-200` folder. The validation set needs to be put into a different folder structure based on classes. We provide a script to do this: `reorg_ti_val.py`.
- TIN597 (available at the OpenOOD repository: https://github.com/Jingkang50/OpenOOD). Download using the code in the repo and place in `data/tin597` folder.

## Training 
We provide scripts for both standard training and training with weight averaging. Each script has a number of hyperparameters that can be adjusted. These include:

- `id_dataset`: The in-distribution dataset to train on. Options are `CIFAR10`, `CIFAR100`, `TinyImageNet`, and - `CIFAR10S` (extra data).
- `batch_size`: The batch size for training.
- `epochs`: The number of epochs to train for.
- `report_freq`: The frequency at which to report training progress.
- `max_lr`: The maximum learning rate for the one-cycle policy.
- `lr`: The learning rate.
- `weight_decay`: The weight decay for SGD.
- `momentum`: The momentum.
- `epsilon`: The perturbation size for adversarial training.
- `beta`: The weight for the KLD loss term.
- `gamma`: The weight for the helper loss term.
- `eta`: The weight for the OE loss term.
- `steps`: The number of steps in the PGD attack.
- `loss_type`: The loss type to use. Options are `STD`, `OE`, `TRADES`, `TRADES_OE`, and `HALO`.
- `model`: The model architecture to use. Options are `ResNet18`, `PreActResNet18` and `TI-PreActResNet18`
- `oe_dataset`: The out-of-distribution dataset to use for OE loss.
- `std_model_path`: The path to the standard model for HALO.
- `device`: The device to train on.
- `seed`: The random seed.
- `run_name`: The name of the run.

To train a model under the standard setting, run the following command:
```
python train.py **args**
```

To train a model with weight averaging, run the following command:
```
python train_wa.py **args**
```

In order to train a model with HALO, you must first train a standard model. Then, you can train a HALO model using the following command:
```
python train.py --loss_type="HALO" --std_model_path="path/to/standard/model.pth" **args**
```

## Evaluation

To evaluate a model, we use a modified version of the OpenOOD framework to allow for adversarial attacks. We provide the following script:
```
python test.py **args**
```
To test a model, pass the `run_name` of the model as an argument along with the `id_dataset`. The script will evaluate the model's accuracy and robust OOD detection performance and produce a json file in the model's directory.

## Acknowledgements
This codebase is based in part on that of [HAT](https://github.com/imrahulr/hat/tree/main?tab=readme-ov-files), [Unlabeled Data Improves Adversarial Robustness](https://github.com/yaircarmon/semisup-adv) and [OpenOOD](https://github.com/Jingkang50/OpenOOD).