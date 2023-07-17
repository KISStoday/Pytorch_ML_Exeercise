import torch
import torch.nn as nn

# block
class block(nn.Module):
    #每个block封了三次卷积操作，对输入通道进行扩展，最后的输出，要对输出通道*4
    #注意，这个块并不进行下采样操作
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        #输入通道先扩展
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #输入和输出相同
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #参考resnet论文里的内容，要对上一层的输出扩展4倍,
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        #下采样
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)


        if self.identity_downsample is not None:
            #下采样, 改变shape
            #这个函数还没有定义？
            identity = self.identity_downsample(identity)
        #下采样之后的特征，进行合并，然后通过RelU
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    # block 是一个List，告诉我们会重复使用上面的残差块多少次  [3, 4, 6, 3]
    # Image_channels 表示特征的通道
    # num_classes告诉我们图像的类别
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        #初始层 init 先对图像进行一次7*7卷积 扩展通道，并采样一次，最大池化
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=3, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        #ResNet laypers  接下来，开始重复执行，每一个stage对应一个layer，每个layer中要完成多次block
        # 此处设置另一个fun， maker layer
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)   #layers[0] : 3
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)  #输出通道是2048

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        #什么时候进行下采样？至于最后一层要下采样，此时步长为1，输出的通道是输入的4背书
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*4))

        #先调用一次残差块，进行处理，提升通道数量
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        #已经计算了一个残差块，所以先减1
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            #256 -> 64, 64 * 4 (256) again，在这个循环结束后，输入和输出肯定是相等的

        #返回一个序列
        return nn.Sequential(*layers)




def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3,8,36,3], img_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2, 3, 244, 244)
    y = net(x).to('cuda')
    print(y.shape)

test()