from main import evaluate
from dataloader import RetinopathyLoader
import torch
import torchvision
from torchvision import transforms
from ResNet import ResNet
from torch.utils.data import DataLoader
import time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    print("Loading model...")
    model = torch.jit.load('model/ResNet50/model(with pretraining).pt', map_location=device)
    print("> Done.")

    data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    print("Loading test data...")
    test_dataset = RetinopathyLoader('data/', 'test', data_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    
    print("Testing model...")
    since = time.time()
    accuracy = evaluate(model, test_dataloader, device, False)
    time_elapsed = time.time() - since
    print("> Done.")
    print(f"Testing time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Accuracy: {accuracy*100:.2f}%")

if __name__ == '__main__':
    print(f"Pytorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    main()
