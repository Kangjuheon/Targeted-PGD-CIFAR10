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
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:35<00:00, 21.73it/s, loss=0.228]
[Epoch 1] Avg Loss: 0.5738
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:35<00:00, 21.93it/s, loss=0.248] 
[Epoch 2] Avg Loss: 0.3808
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:38<00:00, 20.19it/s, loss=0.883] 
[Epoch 3] Avg Loss: 0.3468
Epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:38<00:00, 20.21it/s, loss=0.489] 
[Epoch 4] Avg Loss: 0.3325
Epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:38<00:00, 20.18it/s, loss=0.136] 
[Epoch 5] Avg Loss: 0.3186
Clean Evaluation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:12<00:00,  3.24it/s] 

[Clean Accuracy] 90.24%
PGD Targeted Evaluation: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:18<00:00,  3.46s/it] 

[PGD Targeted Attack Success Rate] eps=0.03, alpha=0.007, iters=10 → 99.97%

```

## Notes
- Target class for each image is chosen randomly and differs from its true label.
- The attack is considered successful when the model predicts the target class.
