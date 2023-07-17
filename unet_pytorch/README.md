![image](https://github.com/KISStoday/Pytorch_ML_Exeercise/assets/38481122/7496c12f-1872-4125-8bb2-f0f9306c0afa)数据集为Kaggle挑战赛Carvana数据集
数据集地址：https://www.kaggle.com/datasets/ipythonx/carvana-image-masking-png
下载后的数据集，需要进行手动切分，自行选定任意数量的图片和标注，作为验证集合。数据需要放在./data文件夹下，格式为：
train_images
train_masks
val_images
val_masks

运行环境：
ubuntu 20.04
python=3.8
pytorch=1.10.1
cuda=11.3
cudnn=8.4.0
albumentations

视频地址：【使用 U-NET 的 PyTorch 图像分割教程】 https://www.bilibili.com/video/BV1Cz4y1J7hP/?share_source=copy_web&vd_source=52bf04ddd6ed585c177519e641c5cc75
代码地址：https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet
