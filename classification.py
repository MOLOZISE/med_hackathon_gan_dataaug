# pip install efficientnet_pytorch

## 학습 코드
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random

from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torchvision import datasets as dt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
# import torchvision

def load_dataloaders(train_data_path, test_data_path="real100gan0_test50"): # train에는 real0gan100 or real100gan0, test에는 real0gan100만
    ## 데이타 로드!!
    batch_size = 64
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    ## make dataset
    #data_path = train_data_path #'real100gan0_train'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
    train_dataset = dt.ImageFolder(
        train_data_path,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    #data_path = test_data_path  # 'real100gan0_train'  # class 별 폴더로 나누어진걸 확 가져와서 라벨도 달아준다
    test_dataset = dt.ImageFolder(
        test_data_path,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))

    ## data split

    train_idx, tmp_idx = train_test_split(list(range(len(train_dataset))), test_size=0.1, random_state=random_seed)
    datasets = {}
    datasets['train'] = Subset(train_dataset, train_idx)
    datasets['valid'] = Subset(train_dataset, tmp_idx)
    datasets['test'] = test_dataset

    ## data loader 선언
    dataloaders, batch_num = {}, {}
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                       batch_size=batch_size, shuffle=True,
                                                       num_workers=2)
    dataloaders['valid'] = torch.utils.data.DataLoader(datasets['valid'],
                                                       batch_size=batch_size, shuffle=False,
                                                       num_workers=2)
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'],
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=2)
    batch_num['train'], batch_num['valid'], batch_num['test'] = len(dataloaders['train']), len(
        dataloaders['valid']), len(dataloaders['test'])
    print('batch_size : %d,  tvt : %d / %d / %d' % (
        batch_size, batch_num['train'], batch_num['valid'], batch_num['test']))

    # num_show_img = 5
    #
    # class_names = {
    #     "0": "no_dr",  # "0": "no_dr"
    #     "1": "mild",  # "1": "mild"
    #     "2": "moderate",  # "2": "moderate"
    #     "3": "severe",  # "3": "severe"
    #     "4": "proliferate"  # "4": "proliferate"
    # }
    #
    # # train check
    # inputs, classes = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # # valid check
    # inputs, classes = next(iter(dataloaders['valid']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])
    # # test check
    # inputs, classes = next(iter(dataloaders['test']))
    # out = torchvision.utils.make_grid(inputs[:num_show_img])  # batch의 이미지를 오려부친다
    # imshow(out, title=[class_names[str(int(x))] for x in classes[:num_show_img]])

    return dataloaders

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, device, criterion, optimizer, num_epochs, dataloaders, model_path="classification_results/"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss, running_corrects, num_cnt = 0.0, 0, 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)

            epoch_loss = float(running_loss / num_cnt)
            epoch_acc = float((running_corrects.double() / num_cnt).cpu() * 100)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
            print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_idx = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('==> best model saved - %d / %.1f' % (best_idx, best_acc))
                model.load_state_dict(best_model_wts)
                torch.save(model.state_dict(), model_path + str(epoch) + ".pt")
                print('model saved')
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best valid Acc: %d - %.1f' % (best_idx, best_acc))
    # load best model weights
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

def draw_graph(train_loss, train_acc, valid_loss, valid_acc, best_idx):
    ## 결과 그래프 그리기
    print('best model : %d - %1.f / %.1f'%(best_idx, valid_acc[best_idx], valid_loss[best_idx]))
    fig, ax1 = plt.subplots()

    ax1.plot(train_acc, 'b-')
    ax1.plot(valid_acc, 'r-')
    plt.plot(best_idx, valid_acc[best_idx], 'ro')
    ax1.set_xlabel('epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('acc', color='k')
    ax1.tick_params('y', colors='k')

    ax2 = ax1.twinx()
    ax2.plot(train_loss, 'g-')
    ax2.plot(valid_loss, 'k-')
    plt.plot(best_idx, valid_loss[best_idx], 'ro')
    ax2.set_ylabel('loss', color='k')
    ax2.tick_params('y', colors='k')

    fig.tight_layout()
    plt.show()

    return 0

def test_and_visualize_model(model, device, dataloaders, phase='test', num_images=4):
    class_names = {
        "0": "no_dr",  # "0": "no_dr"
        "1": "mild",  # "1": "mild"
        "2": "moderate",  # "2": "moderate"
        "3": "severe",  # "3": "severe"
        "4": "proliferate"  # "4": "proliferate"
    }

    # phase = 'train', 'valid', 'test'
    was_training = model.training
    model.eval()
    fig = plt.figure()

    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)  # batch의 평균 loss 출력

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_cnt += inputs.size(0)  # batch size

        #         if i == 2: break

        test_loss = running_loss / num_cnt
        test_acc = running_corrects.double() / num_cnt
        print('test done : loss/acc : %.2f / %.1f' % (test_loss, test_acc * 100))

    # 예시 그림 plot
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 예시 그림 plot
            for j in range(1, num_images + 1):
                ax = plt.subplot(num_images // 2, 2, j)
                ax.axis('off')
                ax.set_title('%s : %s -> %s' % (
                    'True' if class_names[str(labels[j].cpu().numpy())] == class_names[
                        str(preds[j].cpu().numpy())] else 'False',
                    class_names[str(labels[j].cpu().numpy())], class_names[str(preds[j].cpu().numpy())]))
                imshow(inputs.cpu().data[j])
            if i == 0: break

    model.train(mode=was_training);  # 다시 train모드로

if __name__ == "__main__":
    model_name = 'efficientnet-b0'  # b5
    model = EfficientNet.from_pretrained(model_name, num_classes=5)

    #dataloaders = load_dataloaders(train_data_path="real0gan100_train", test_data_path="real100gan0_test50")
    #dataloaders = load_dataloaders(train_data_path="real100gan0_train", test_data_path="real100gan0_test50")
    dataloaders = load_dataloaders(train_data_path="real0gan100_train_5000", test_data_path="real100gan0_test50")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    ### 모델 훈련
    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(model, device, criterion, optimizer_ft, num_epochs=20, dataloaders=dataloaders)
    ### 훈련 그래프
    draw_graph(train_loss, train_acc, valid_loss, valid_acc, best_idx)
    ###
    test_and_visualize_model(model, device, dataloaders=dataloaders, phase='test', num_images=4)

