import torch
def evaluate_model(model, data_loader, criterion, device='cpu'):
_correct = 0
_total = 0
with torch.no_grad():
    for _data in test_loader: # change to data_loader
        _images, _labels = data
        _outputs = model(_images)
        _, _predicted = torch.max(_outputs.data, 1)
        _total += _labels.size(0)
        _correct += (_predicted == _labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f}%")

return avg_loss, accuracy

