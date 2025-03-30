# PGD Targeted Attack on CIFAR-10 (ConvNeXt)

This project implements a **targeted PGD (Projected Gradient Descent)** adversarial attack against a fine-tuned **ConvNeXt-Tiny** model on CIFAR-10.

## Overview

- Model: ConvNeXt-Tiny pretrained on ImageNet
- Training: Only the final classifier layer trained
- Attack: Targeted PGD with multiple steps and random targets
- Evaluation: Success rate of pushing predictions toward target classes

## Files

- `test.py`: Runs the training and targeted PGD attack
- `requirements.txt`: Dependencies

## Run Instructions

```bash
pip install -r requirements.txt
python test.py
```

## Example Output
```bash


```

## Notes
- Target class for each image is chosen randomly and differs from its true label.
- The attack is considered successful when the model predicts the target class.
