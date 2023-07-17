import  torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  #加载进度
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,

)
#Hyperparameters etc.

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #检查cuda是否可用
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240   # 1918
PIN_MEMORY = True
LOAD_MODEL = False # True
# TRAIN_IMG_DIR = "./data/train_images/"
# TRAIN_MASK_DIR = "./data/train_masks/"
# VAL_IMG_DIR = "./data/val_images/"
# VAL_MASK_DIR = "./data/val_masks/"

TRAIN_IMG_DIR = ".\\data\\train_images\\"
TRAIN_MASK_DIR = ".\\data\\train_masks\\"
VAL_IMG_DIR = ".\\data\\val_images\\"
VAL_MASK_DIR = ".\\data\\val_masks\\"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    #pass  #是一个空语句,为了保持程序结构的完整性
    loop = tqdm(loader)  #tqdm是什么，可以看他的视频

    for batch_idx, (data,targets), in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, targets)

        #backward
        optimizer.zero_grad()
        scaler.scsale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),  #垂直归一化
            A.VerticalFlip(p=0.1),      #水平归一化
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,  #像素值除以255，得到0-1之间的一个数值
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  #这里只做二分类，所以最终的输出通道就是1
    #如果想做多分类，就改成cross entropy loss，把输出通道改成3，代表rgb

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )  #之后要创建出需要的util.py文件

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),

        }
        save_checkpoint(checkpoint)

        #check accuracy

        check_accuracy(val_loader, model, device=DEVICE)

        #print some examples to a folder
        save_predictions_as_imgs(
            # val_loader, model, folder="saved_images/", device=DEVICE
            val_loader, model, folder="C:\\Users\\79413\\OneDrive\\ml_exercise\\unet_pytorch\\data\\saved_images\\", device=DEVICE
        )





if __name__ == "__main__":
    main()



