# -*- coding: utf-8 -*-


import os
from PIL import Image 

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models 

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



# 学習データ、検証データへの分割
def make_filepath_list():
    
    # 学習データ、検証データそれぞれのファイルへのパスを格納したリストを返す
    train_file_list = []
    valid_file_list = []

    imagedir = '/Users/sakaitaigoware/Documents/pytorch/fish_images/'

    for top_dir in os.listdir(imagedir):
        file_dir = os.path.join(imagedir, top_dir)
        file_list = os.listdir(file_dir)

        # 学習データ40,検証データ10とする
        num_data = len(file_list)
        num_split = int(num_data - 10)

        # 以下で'\\'を'/'にreplaceしているのはWindowsでのpath出力に対応するため
        train_file_list += [os.path.join(imagedir, top_dir, file).replace('\\', '/') for file in file_list[:num_split]]
        valid_file_list += [os.path.join(imagedir, top_dir, file).replace('\\', '/') for file in file_list[num_split:]]
    
    return train_file_list, valid_file_list

# 前処理クラス
class ImageTransform(object):
    """
    resize: int
        リサイズ先の画像の大きさ
    mean: (R, G, B)
        各色チャンネルの平均値
    std: (R, G, B)
        各色チャンネルの標準偏差
    """
    def __init__(self, resize, mean, std):
        # 辞書型でMethodを定義
        self.data_trasnform = {
            'train': transforms.Compose([
                # データオーグメンテーション
                transforms.RandomHorizontalFlip(),
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ]),
            'valid': transforms.Compose([
                # 画像をresize×resizeの大きさに統一する
                transforms.Resize((resize, resize)),
                # Tensor型に変換する
                transforms.ToTensor(),
                # 色情報の標準化をする
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_trasnform[phase](img)
    
# Datasetクラス
class Dataset(data.Dataset):
    """    
    file_list: list
        画像のファイルパスを格納したリスト
    classes: list
        ラベル名
    transform: object
        前処理クラスのインスタンス
    phase: 'train' or 'valid'
        学習か検証化を設定
    """
    def __init__(self, file_list, classes, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.phase = phase
    
    def __len__(self):
        """
        画像の枚数を返す
        """
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        前処理した画像データのTensor形式のデータとラベルを取得
        """
        # 指定したindexの画像を読み込む
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('RGB')
        
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)
        
        # 画像ラベルをファイル名から抜き出す
        # /Users/sakaitaigoware/Documents/pytorch/fish_images/の後ろの文字列を抜き出す
        label = self.file_list[index].split('/')[6][:] # '/'で分けた5番目

        # ラベル名を数値に変換
        label = self.classes.index(label)
        
        return img_transformed, label


if __name__ == '__main__':

    time_start = time.time()

    #再現性を保つためにseedを固定
    seed = 11
    #random.seed(seed)
    #np.random.seed(seed)  
    torch.manual_seed(seed) 

    # 各種パラメータの用意
    # クラス名
    myclasses = [
        'fish',  'goldfish',
    ]

    # リサイズ先の画像サイズ
    resize = 300

    # mean = (0.549, 0.494, 0.44)
    # std = (0.262, 0.239, 0.246)

    mean = (0.549, 0.494, 0.44)
    std = (0.262, 0.239, 0.246)

    # バッチサイズの指定
    batch_size = 10  

    # エポック数
    num_epochs = 1

    # GPU使用を試みる
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device,'\n')

    # 2. 前処理
    # 学習データ、検証データのファイルパスを格納したリストを取得する
    train_file_list, valid_file_list = make_filepath_list()
    # 3. Datasetの作成
    train_dataset = Dataset(
        file_list=train_file_list, classes=myclasses,
        transform=ImageTransform(resize, mean, std),
        phase='train'
    )
    valid_dataset = Dataset(
        file_list=valid_file_list, classes=myclasses,
        transform=ImageTransform(resize, mean, std),
        phase='valid'
    )
    # 4. DataLoaderの作成
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=32, shuffle=False)
    # 辞書にまとめる
    dataloaders_dict = {
        'train': train_dataloader, 
        'valid': valid_dataloader
    }
    # 5. ネットワークの定義
    net = models.alexnet(pretrained = True)
    net.fc = nn.Linear(in_features=4096, out_features=2)
    net = net.to(device)                    
    # 6. 損失関数の定義
    criterion = nn.CrossEntropyLoss()
    # 7. 最適化手法の定義
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    # 8. 学習・検証
    epoch_losses = np.empty(0, dtype=float)
    epoch_accs = np.empty(0, dtype=float)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        for phase in ['train', 'valid']:
            if phase == 'train':
                # 学習モードに設定
                net.train()
            else:
                # 訓練モードに設定
                net.eval()
                
            # epochの損失和
            epoch_loss = 0.0
            # epochの正解数
            epoch_corrects = 0.0
            
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # optimizerを初期化
                optimizer.zero_grad()
                
                # 学習時のみ勾配を計算させる設定にする
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = net(inputs)
                    
                    # 損失を計算
                    loss = criterion(outputs, labels)
                    
                    # ラベルを予測
                    _, preds = torch.max(outputs, 1)
                    
                    # 訓練時は逆伝搬の計算
                    if phase == 'train':
                        # 逆伝搬の計算
                        loss.backward()
                        
                        # パラメータ更新
                        optimizer.step()
                        
                    epoch_loss += loss.item()
                    
                    epoch_corrects += torch.sum(preds == labels.data)
                    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                    epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
                    
                    # 一時的保存
                    _epoch_loss = epoch_loss
                    _epoch_acc = epoch_acc.item()

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # 1 epoch
        epoch_losses = np.append(epoch_losses, _epoch_loss)
        epoch_accs = np.append(epoch_accs, _epoch_acc)

        time_end = time.time()
        print('\nElapsed time: {:.3f} sec'.format(time_end - time_start))

    # 5 epoch 
    x = np.arange(1, num_epochs + 1, 1)
    plt.figure(figsize=(9,8))
    plt.plot(x, epoch_losses, label = "Loss")
    plt.plot(x, epoch_accs, linestyle="--", label = "Accracy")
    plt.legend()
    plt.show()

    # モデルを保存、確認
    PATH = './alex_net.pth'
    torch.save(net, PATH)
    net = torch.load(PATH)
    net.eval()

    # 動作確認
    num_photo = 50

    idir_1 = '/Users/sakaitaigoware/Documents/pytorch/fish_images/fish/'
    idir_2 = '/Users/sakaitaigoware/Documents/pytorch/fish_images/goldfish/'
    def detect(num, idir):
        for k in range(num): 
            if k < 9:
                img = cv2.imread(idir + "0000" + str(k+1) + '.jpg')
                print(idir + "0000" + str(k+1) + '.jpg')
                img = Image.open(idir + "0000" + str(k+1) + '.jpg')
            else:
                img = cv2.imread(idir + "000" + str(k+1) + '.jpg')
                print(idir + "000" + str(k+1) + '.jpg')
                img = Image.open(idir + "000" + str(k+1) + '.jpg')

            transform=ImageTransform(resize, mean, std)
            try:
                img_valid = transform(img, 'train')
            except RuntimeError:
                print("error")
                continue
            # plt.imshow(img)
            # plt.show()

            net.eval()
            img_valid = img_valid.unsqueeze(0).to(device)
            out = net(img_valid)
            # ラベルを求める
            _, preds = torch.max(out, 1)


            print('Predicted label:', myclasses[preds.item()])
            
    #関数
    detect(num_photo, idir_1)
    detect(num_photo, idir_2)

