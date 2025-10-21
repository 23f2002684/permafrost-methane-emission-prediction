import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
#Configuration
DATA_DIR = 'data'
MODEL_SAVE_PATH = 'models/methane_classifier_final_500plus.pth' 
NUM_CLASSES = 3
BATCH_SIZE = 16
NUM_EPOCHS = 35
LEARNING_RATE = 0.001
IMG_SIZE = 224
#Custom Dataset for PNGs
class PngDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.label_map = {"LowRisk": 0, "ModerateRisk": 1, "HighRisk": 2}
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        try:
            image = Image.open(filepath).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label_name = os.path.basename(os.path.dirname(filepath))
            label = self.label_map[label_name]
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            # Return a dummy tensor and label or skip this sample
            # This handles potential corrupted images in a large dataset
            dummy_image = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_image, dummy_label
#TRAINING
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_files, all_labels = [], []
    class_names = ["LowRisk", "ModerateRisk", "HighRisk"]
    for risk_level in class_names:
        risk_path = os.path.join(DATA_DIR, risk_level)
        if os.path.isdir(risk_path):
            count = 0
            for filename in os.listdir(risk_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(risk_path, filename))
                    all_labels.append(risk_level)
                    count += 1
            print(f"Found {count} images in {risk_path}")
        else:
             print(f"Warning: Directory not found {risk_path}")
    if not all_files:
        print("Error: No image files found in the 'data' directory.")
        exit()
    print(f"Total images found: {len(all_files)}")
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    train_dataset = PngDataset(train_files, train_labels, transform=transform)
    # Use the validation transform for the validation set
    val_dataset = PngDataset(val_files, val_labels, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) 
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    # Lists to store metrics for plotting
    train_losses, val_losses, val_accuracies = [], [], []
    print(f"Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader): 
            # Skip potential bad samples from dataset error handling
            if torch.equal(inputs, torch.zeros((inputs.size(0), 3, IMG_SIZE, IMG_SIZE))):
                print(f"Skipping potentially corrupted batch at index {i}")
                continue   
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                 # Skip potential bad samples
                if torch.equal(inputs, torch.zeros((inputs.size(0), 3, IMG_SIZE, IMG_SIZE))):
                    continue      
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total > 0:
             val_loss /= len(val_loader)
             accuracy = 100 * correct / total
        else:
             val_loss = 0.0
             accuracy = 0.0
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        scheduler.step() 
    print("Finished Training.")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    #PLOTTING LEARNING CURVES
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(train_losses, label='Training Loss', color='royalblue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='darkorange', linewidth=2)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Model Loss Curves", fontsize=14)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='forestgreen', linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.set_title("Model Accuracy Curve", fontsize=14)
    fig.suptitle('Training and Validation Metrics', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('learning_curves.png', dpi=300)
    print("Learning curve graph saved to learning_curves.png")