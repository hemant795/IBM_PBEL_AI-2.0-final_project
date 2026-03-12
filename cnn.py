import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# ── Device Setup ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU Name:         {torch.cuda.get_device_name(0)}')
    print(f'Allocated Memory: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB')
    print(f'Cached Memory:    {torch.cuda.memory_reserved(device)  / 1024**3:.1f} GB')

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_set = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=100, shuffle=False, num_workers=0)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1      = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2      = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1        = nn.Linear(320, 50)
        self.fc2        = nn.Linear(50, 10)
        self.drop       = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x  # Raw logits — CrossEntropyLoss handles softmax internally

model     = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

# Training
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            allocated = torch.cuda.memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0
            cached    = torch.cuda.memory_reserved(device)  / 1024**3 if device.type == 'cuda' else 0
            print(f'Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}'
                  f'\tGPU Alloc: {allocated:.3f} GB | Cached: {cached:.3f} GB')

# Testing
def test():
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output     = model(data)
            test_loss += loss_fn(output, target).item()
            pred       = output.argmax(dim=1, keepdim=True)
            correct   += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg loss: {test_loss:.4f} | '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy

# Run Training Loop
print('\n--- Training CNN ---')
best_accuracy = 0
for epoch in range(1, 11):
    train(epoch)
    acc = test()
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  ✅ New best model saved ({best_accuracy:.1f}%)')

if device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f'Post-training GPU — '
          f'Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.3f} GB | '
          f'Cached: {torch.cuda.memory_reserved(device) / 1024**3:.3f} GB\n')

# Prediction Grid
print('--- Showing prediction grid ---')
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('CNN Predictions on MNIST Test Set — Green: Correct | Red: Wrong', fontsize=13)

for ax in axes.flat:
    idx = random.randint(0, len(test_set) - 1)
    data, target = test_set[idx]
    with torch.no_grad():
        output     = model(data.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).item()
    ax.imshow(data.squeeze().numpy(), cmap='gray')
    color = 'green' if prediction == target else 'red'
    ax.set_title(f'Pred: {prediction}  |  True: {target}', color=color, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()