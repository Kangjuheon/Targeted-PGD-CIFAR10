import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PGD Targeted Attack
def pgd_targeted_attack(model, x, target, eps, alpha, iters):
    x_adv = x.clone().detach().to(device)
    x_adv = x_adv + 0.001 * torch.randn_like(x_adv)  # random start
    x_adv = x_adv.detach()
    x_adv.requires_grad = True

    for _ in range(iters):
        output = model(x_adv)
        loss = F.cross_entropy(output, target.to(device))
        model.zero_grad()
        loss.backward()
        x_adv = x_adv - alpha * x_adv.grad.data.sign()  # ← targeted: 방향 반대
        eta = torch.clamp(x_adv - x, min=-eps, max=eps)
        x_adv = torch.clamp(x + eta, 0, 1).detach()
        x_adv.requires_grad = True
    return x_adv

# 랜덤 목표 클래스 생성
def generate_random_targets(labels, num_classes=10):
    return torch.tensor([(label + random.randint(1, 9)) % num_classes for label in labels])

# Clean Accuracy
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in tqdm(loader, desc="Clean Evaluation"):
            x, label = x.to(device), label.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += pred.eq(label).sum().item()
            total += len(x)
    return 100 * correct / total

# Targeted PGD 공격 성공률
def evaluate_targeted_attack(model, loader, eps, alpha, iters):
    model.eval()
    success = 0
    total = 0
    for x, label in tqdm(loader, desc="PGD Targeted Evaluation"):
        x, label = x.to(device), label.to(device)
        target = generate_random_targets(label).to(device)
        x_adv = pgd_targeted_attack(model, x, target, eps, alpha, iters)
        with torch.no_grad():
            output = model(x_adv)
            pred = output.argmax(dim=1)
            success += pred.eq(target).sum().item()
            total += len(x)
        torch.cuda.empty_cache()
        gc.collect()
    return 100 * success / total

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
    model = model.to(device)

    # Train classifier
    optimizer = optim.Adam(model.classifier[2].parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, label in loop:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"[Epoch {epoch+1}] Avg Loss: {total_loss / len(train_loader):.4f}")

    clean_acc = evaluate(model, test_loader)
    print(f"\n[Clean Accuracy] {clean_acc:.2f}%")

    # PGD Targeted Attack
    eps = 0.03
    alpha = 0.007
    iters = 10
    success = evaluate_targeted_attack(model, test_loader, eps, alpha, iters)
    print(f"\n[PGD Targeted Attack Success Rate] eps={eps}, alpha={alpha}, iters={iters} → {success:.2f}%")
