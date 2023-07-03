import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.optim as optim
import multiprocessing
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Define the classes you want to use
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example: Selecting the first 10 classes

    # Define the data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the CIFAR-100 training dataset and filter the classes
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, [idx for idx, label in enumerate(trainset.targets) if label in selected_classes])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)

    # Load the pre-trained MobileNetV2 model
    model = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

    # Modify the classifier for 10 classes
    num_classes = 10
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Training loop
    for epoch in tqdm(range(0,100)):
        running_loss = 0.0
        model.to(device)
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss:.3f}')

    print(f'Loss: {running_loss:.3f}')
    # Save the trained model
    torch.save(model.state_dict(), 'mobilenet_model.pth')
