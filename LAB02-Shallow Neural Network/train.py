import math
import torch
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from models import MyModel
from dataset import MLDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# validation package
#import torch.utils.data as dataf


# WRMSE
def WRMSE(preds, labels, device):
    weight = torch.tensor([
        0.05223, 0.0506, 0.05231, 0.05063, 0.05073,
        0.05227, 0.05177, 0.05186, 0.05076, 0.05063,
        0.0173, 0.05233, 0.05227, 0.05257, 0.05259,
        0.05222, 0.05204, 0.05185, 0.05229, 0.05074
    ]).to(device)
    wrmse = torch.pow(preds-labels, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()

# training curve
def visualize(record, title,record2,title2):
    plt.title(title+" "+title2)
    # plt.title(title2)
    # plt.plot(record)
    # plt.gca().set_xlim([0, 30])
    # plt.gca().set_ylim([0, 0.05])
    train, = plt.plot(record, label="train")
    val, = plt.plot(record2, label="train")


    # plt.plot(record,'b', label='train_loss')
    # plt.plot(record2,'r', label = 'val_loss' )
    plt.legend([train, val], ["train_loss", "val_loss"], loc='best')
    plt.show()

# learning rate, epoch and batch size. Can change the parameters here.
def train(lr=0.001, epoch=30, batch_size=32):
    train_loss_curve = []
    train_wrmse_curve = []
    #*****************append validation loos curve and validation wrmse curve *****
    validation_loss_curve = []
    validation_wrmse_curve = []
    #******************************************************************************

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()
    model.eval()
# radaom data set
#     train_set = datasets.MLDataset()
#     test_set = datasets.MLDataset()
#     train_set_size = int(len(train_set) * 0.8)
#     valid_set_size = len(train_set) - train_set_size
#     train_set, valid_set = val_data.random_split(train_set, [train_set_size, valid_set_size])
#

    # dataset and dataloader
    # can use torch random_split to create the validation dataset
    dataset = MLDataset()
    #********************************************************************************
    #here sparate train_data & test_data
    elements = list(dataset)
    train_dataloader, test_dataloader = train_test_split(elements, train_size=0.9)
    print('train:',len(train_dataloader),'test:',len(test_dataloader))


    train_dataloader = DataLoader(dataset=train_dataloader, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataloader, batch_size=batch_size, shuffle=True)
    # print('Train: {} Test: {}'.format( len(train_dataloader), len(test_dataloader)) )
    #********************************************************************************

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    for e in range(epoch):
        train_loss = 0.0
        train_wrmse = 0.0
        valid_loss = 0.0
        valid_wrmse = 0.0
        best = 100
        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs, labels in tqdm(train_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)

            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss calculate
            train_loss += loss.item()
            train_wrmse += wrmse
        # =================================================================== #
        # If you have created the validation dataset,
        # you can refer to the for loop above and calculate the validation loss
        for inputs, labels in tqdm(test_dataloader):
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            outputs = model(inputs)
            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)
            # loss calculate
            valid_loss += loss.item()
            valid_wrmse += wrmse


        # =================================================================== #
        # save the best model weights as .pth file
        loss_epoch = train_loss / len(train_dataloader.dataset)
        valid_loss_epoch = valid_loss / len(test_dataloader.dataset)

        wrmse_epoch = math.sqrt(train_wrmse/len(train_dataloader.dataset))
        valid_wrmse_epoch = math.sqrt(valid_wrmse/len(test_dataloader.dataset))

        if wrmse_epoch < best :
            best = wrmse_epoch
            torch.save(model.state_dict(), 'mymodel.pth')
        print(f'Training loss: {loss_epoch:.4f}')
        print(f'Training WRMSE: {wrmse_epoch:.4f}')
        print(f'Validation loss: {valid_loss_epoch:.4f}')
        print(f'Validation WRMSE: {valid_wrmse_epoch:.4f}')
        # save loss and wrmse every epoch
        train_loss_curve.append(loss_epoch)
        train_wrmse_curve.append(wrmse_epoch)
        validation_loss_curve.append(valid_loss_epoch)
        validation_wrmse_curve.append(valid_wrmse_epoch)
    # generate training curve
    print(train_wrmse_curve)
    print(validation_wrmse_curve)
    visualize(train_loss_curve, 'Train Loss', train_wrmse_curve, 'Validation Loss')
    visualize(validation_loss_curve, 'Train WRMSE', validation_wrmse_curve, 'Validation WRMSE')

if __name__ == '__main__':
    train()