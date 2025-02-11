import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pool = nn.MaxPool2d(2, 2, 1, 2)

        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm1d(120)
        self.bn6 = nn.BatchNorm1d(84)

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print("After conv1 and pool:", x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print("After conv2 and pool:", x.shape)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print("After conv3 and pool:", x.shape)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print("After conv4 and pool:", x.shape)
        x = x.view(-1, 64 * 4 * 4)
        # print("After view:", x.shape)
        x = F.relu(self.bn5(self.fc1(x)))
        # print("After fc1:", x.shape)
        x = F.relu(self.bn6(self.fc2(x)))
        # print("After fc2:", x.shape)
        x = self.fc3(x)
        # print("Output shape:", x.shape)
        return x


def train(net, train_batches, val_batches, epochs, learning_rate, device, momentum):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()

    for _ in range(epochs):
        for batch in train_batches:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    val_loss, val_acc = test(net, val_batches, device)

    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, test_batch, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in test_batch:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(test_batch.dataset)
    loss = loss / len(test_batch.dataset)
    return loss, accuracy
