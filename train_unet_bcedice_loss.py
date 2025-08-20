import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = []
        for img in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img)
            mask_path = os.path.join(mask_dir, img)
            if os.path.exists(mask_path) and self._is_valid_image(img_path) and self._is_valid_image(mask_path):
                self.images.append(img)

        if len(self.images) == 0:
            raise RuntimeError(f"No valid image-mask pairs found in {image_dir} and {mask_dir}")

    def _is_valid_image(self, path):
        try:
            Image.open(path).verify()
            return True
        except:
            return False

    def __len__(self):
        return len(self.images)

    def read_image(self, path, is_mask=False):
        img = Image.open(path)
        return img.convert("L") if is_mask else img.convert("RGB")

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = self.read_image(img_path, is_mask=False)
        mask = self.read_image(mask_path, is_mask=True)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()  # Binarize mask

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


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return 1 - dice.mean()

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss_val = dice_loss(pred, target)
        return bce_loss + dice_loss_val


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(
    'data/images',
    'data/masks',
    transform_img=transform_img,
    transform_mask=transform_mask
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = UNet(in_channels=3, out_channels=1).to(device)
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch [{epoch+1}/50], Loss: {epoch_loss/len(train_loader):.4f}")

os.makedirs("backend/model", exist_ok=True)
torch.save(model.state_dict(), "backend/model/unet_permafrost.pth")
