import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
#Configuration
DATA_DIR = 'data'
MODEL_SAVE_PATH = 'models/methane_classifier_final.pth'
NUM_CLASSES = 3
BATCH_SIZE = 16 
NUM_EPOCHS = 40 
LEARNING_RATE = 0.001
IMG_SIZE = 224
#PNG Dataset
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
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_name = os.path.basename(os.path.dirname(filepath))
        label = self.label_map[label_name]
        return image, torch.tensor(label, dtype=torch.long)
if __name__ == '__main__':
    # More aggressive data augmentation
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_files, all_labels = [], []
    for risk_level in ["LowRisk", "ModerateRisk", "HighRisk"]: # Ensure correct order
        risk_path = os.path.join(DATA_DIR, risk_level)
        if os.path.isdir(risk_path):
            for filename in os.listdir(risk_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append(os.path.join(risk_path, filename))
                    all_labels.append(risk_level)
    
    if not all_files:
        print("Error: No PNG files found in the 'data' directory.")
        exit()

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_dataset = PngDataset(train_files, train_labels, transform=transform)
    val_dataset = PngDataset(val_files, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #MODEL
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # --- ADD LEARNING RATE SCHEDULER ---
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    print(f"Starting training on {len(all_files)} images...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        #VALIDATION LOOP
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {accuracy:.2f}%")
        scheduler.step()
    print("Finished Training :)")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")