import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import resnet
import json
from main import compare_models

def main():

    train_losses = [ ]
    val_losses = [ ]
    train_accuracies = [ ]
    val_accuracies = [ ]

    coefficients = []

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the pre-trained model
    model = resnet.resnet34()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('resnet34_weights.pth'))

    # Load model as a comparison
    compare_model = resnet.resnet34()
    compare_model = torch.nn.DataParallel(compare_model)
    compare_model.load_state_dict(torch.load('resnet34_weights.pth'))

    model = model.to(device)
    compare_model = compare_model.to(device)

    # Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the ExponentialLR scheduler
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Load in values for comparison
    with open('config.json', 'r') as f:
        config = json.load(f)

    bit_to_change = config[ "bit_to_change" ]
    first_random_number = config[ "first_random_number" ]
    second_random_number = config[ "second_random_number" ]
    third_random_number = config[ "third_random_number" ]

    num_epochs = 20  # Set the number of epochs

    coeff = compare_models(compare_model, model, bit_to_change, first_random_number, second_random_number, third_random_number)
    coefficients.append(coeff)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Step the scheduler at the end of each epoch
        #scheduler.step()
        avg_train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(testloader)
        accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.2f}%')

        coeff = compare_models(compare_model, model, bit_to_change, first_random_number, second_random_number, third_random_number)
        coefficients.append(coeff)

    # After training, plot the Pearson correlation coefficients
    plt.plot(range(0, num_epochs + 1), coefficients, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Pearson Correlation Coefficient over Epochs')
    plt.show()

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_model.pth')

    # Plot for training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot for training and validation accuracy
    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
