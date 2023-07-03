
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.optim as optim
import multiprocessing

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

    # Load the CIFAR-100 test dataset and filter the classes
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, [idx for idx, label in enumerate(testset.targets) if label in selected_classes])
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Load the saved model
    model = mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

    # Modify the classifier for 10 classes
    num_classes = 10
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(torch.load('mobilenet_model.pth'))
    model = model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

