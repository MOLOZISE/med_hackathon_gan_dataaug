# med_hackathon_gan_dataaug

Using FastGAN and adding other ViT based modules.

Because of time limit, not much moderate on models.

In this project I try to generate images with small dataset in Diabetic Retinopathy task(Vision Classification Task)

There are 5 stages in Diabetic Retinopathy(DR)

![20220828_01](https://user-images.githubusercontent.com/83456681/190573416-09dab412-f7e3-4484-995c-802e23fa83da.png)

I'm trying to train fast as I can for time limitation, so using FastGAN was inevitable.

And moderating with MaxViT's modules(MBConv block, Block Attention, Grid Attention)

Using only 500 each on DR datasets to train moderated models with 50000 epoches, Almost 7 ~ 10 hours to complete train.

0 epoches
![0](https://user-images.githubusercontent.com/83456681/190574375-593309ba-e0a0-4b73-9e33-c0544eea8583.jpg)
10000 epoches
![10000](https://user-images.githubusercontent.com/83456681/190574382-0bd78ee3-9df1-4b59-8ef1-75bd29cf5fd5.jpg)
20000 epoches
![20000](https://user-images.githubusercontent.com/83456681/190574404-42fb1f72-e02b-4c4f-8057-c96c536a6302.jpg)
30000 epoches
![30000](https://user-images.githubusercontent.com/83456681/190574418-9f9d8b57-70c9-44ea-a46d-c43911f1632c.jpg)
40000 epoches
![40000](https://user-images.githubusercontent.com/83456681/190574431-299bacda-48a2-42fd-bce1-03b81dd56376.jpg)
50000 epoches
![50000](https://user-images.githubusercontent.com/83456681/190574440-9b34f211-fa51-45ad-8963-668d656093a9.jpg)

Look clearly learned well, but on classification, there was some overfitting on classifcation model(using pre-trained efficient-net)

As a result, It was bad for using only GAN images for training, because of overfitting.

To generate better train data, I think focus on few-shot learning on GAN + more stochastic modules can helf overcome overfitting bias in image generation by GAN.


FastGAN + MaxViT, ViTGAN
