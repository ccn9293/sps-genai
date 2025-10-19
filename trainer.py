import torch
from tqdm import tqdm
from .checkpoints import save_checkpoint

EPOCHS = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

correct = 0
total = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1}, {i + 1}] accuracy: {correct/total:.3f}, loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print("Finished Training")


correct = 0
total = 0
with torch.no_grad():
    for _data in test_loader:
        _images, _labels = data
        _outputs = model(_images)
        _, _predicted = torch.max(_outputs.data, 1)
        _total += _labels.size(0)
        _correct += (_predicted == _labels).sum().item()

accuracy = 100 * correct / total
loss= 1- accuracy
print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")

    

os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
save_checkpoint(model, optimizer, epoch, train_loss, train_accuracy, checkpoint_path)


if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)


