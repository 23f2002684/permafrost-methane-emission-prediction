import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = [img for img in os.listdir(image_dir) if os.path.exists(os.path.join(mask_dir, img))]
        if not self.images:
            raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()

        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d1 = torch.cat([self.up1(e4), e3], dim=1)
        d1 = self.dec1(d1)
        d2 = torch.cat([self.up2(d1), e2], dim=1)
        d2 = self.dec2(d2)
        d3 = torch.cat([self.up3(d2), e1], dim=1)
        d3 = self.dec3(d3)
        return torch.sigmoid(self.final(d3))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

#Setup Training Parameters and Paths
IMAGE_DIR = '/content/drive/MyDrive/MLprojcore/data/images/'
MASK_DIR = '/content/drive/MyDrive/MLprojcore/data/masks/'
MODEL_SAVE_PATH_DIR = '/content/drive/MyDrive/MLprojcore/model/'
os.makedirs(MODEL_SAVE_PATH_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_PATH_DIR, 'unet_permafrost_focal_loss.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform_img = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
transform_mask = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_dataset = CustomDataset(IMAGE_DIR, MASK_DIR, transform_img=transform_img, transform_mask=transform_mask)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

#Initialize Model and Start Training with Focal Loss
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("\nStarting model training with Focal Loss...")

for epoch in range(50):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{50}", leave=True)

    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/50] completed. Average Loss: {avg_epoch_loss:.4f}")

print("\nTraining finished!")

#Save the New Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel trained with Focal Loss saved to: {MODEL_SAVE_PATH}")
