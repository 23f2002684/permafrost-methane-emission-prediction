import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models # Import models
from PIL import Image
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = 'data'
MODEL_SAVE_PATH = 'models/methane_classifier_png.pth'
NUM_CLASSES = 3  # High, Low, Moderate
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IMG_SIZE = 224

# --- Custom Dataset for PNGs ---
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

# --- Main Training Logic ---
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_files, all_labels = [], []
    for risk_level in os.listdir(DATA_DIR):
        risk_path = os.path.join(DATA_DIR, risk_level)
        if os.path.isdir(risk_path):
            for filename in os.listdir(risk_path):
                if filename.lower().endswith('.png'):
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
    
    # --- FIXED MODEL LOADING ---
    # Load ResNet18 with the recommended weights API
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the final layer for our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training on PNG images... 🚀")
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
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

    print("✅ Finished Training.")
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")