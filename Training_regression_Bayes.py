import torch.optim as optim
from torch.utils.data.dataset import Subset
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import random
import seaborn as sn

import torch
from Unet_regression import UNet
from FNC_regression import FNC
from mydataset import Dataset_reg as Dataset
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.autograd import Variable
import pandas as pd
from BayesModel import ResNext
from Losses_NLL import negative_log_likelihood

from utils import nanmean, limit
s=34

learning_rate = 0.1
num_epochs = 100
batch_size = 8
size='1920_crop_'

m='RexNext_biomass_Regression'
channel=6
name=str(m)+str(size)+'_z+'+str(num_epochs)+'ep_s_'+str(batch_size)+'_WeightTest1_Albu_wd_10-5_channel_'+str(channel)
print(name)



cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FNC(n_classes=1, in_channels=6).to(device)
# model = UNet(n_classes=1, padding=True, up_mode='upconv', in_channels=channel).to(device)
model = ResNext(channel, 1, layers= [2,3,5,3], groups= 32, width_per_group= 4).to(device)



train_dataset = Dataset('/home/beland/PycharmProjects/Data_biomass/IMAGETTES_'+size+'/tilefiles/TRAIN.txt', augment=False)
print("train_dataset", len(train_dataset))
test_dataset = Dataset('/home/beland/PycharmProjects/Data_biomass/IMAGETTES_'+size+'/tilefiles/TEST.txt', augment=False)

print("test_dataset", len(test_dataset))
valid_dataset = Dataset('/home/beland/PycharmProjects/Data_biomass/IMAGETTES_'+size+'/tilefiles/VALID.txt', augment=False)

print("valid_dataset", len(valid_dataset))

print("fer",len(train_dataset))





train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
print("valid", len(valid_dataset))

dataloader = train_loader, test_loader




#Horaire
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epoch_list, gamma=0.5, last_epoch=-1)

def weighted_mse_loss(input,target):

    w=target
    w[torch.where(w==0)] = 1
    w[torch.where((w > 0)&(w<=50))] = 1
    w[torch.where((w > 50)&(w<=150))] = 1
    w[torch.where((w > 150))] = 1
    weights = w

    return ((input - target) ** 2).mean()

def custom_mean_square_error(y_predictions, target):
    w=target.detach().clone()
    w[torch.where(w==0)] = 0.5
    w[torch.where((w > 0)&(w<=50))] = 2
    w[torch.where((w > 50)&(w<=150))] = 1
    w[torch.where((w > 150))] = 3

    square_difference = torch.square(y_predictions - target)

    loss_value = torch.mean(square_difference)
    loss = (w*(y_predictions - target) ** 2).mean()

    return loss


def Regression_accuracy(y_pred, y_true):
    #Calcul le nombre de bonne réponse en fonction d'un pourcentage d'approximation (pct_close)
    pct_close = 5
    n_items = (y_true.shape)
    n_correct= torch.sum((torch.abs(y_pred - y_true)) < (pct_close ))
    #print(((torch.abs(y_pred - y_true)) < torch.abs(pct_close * y_true)))
    #n_correct = torch.sum((torch.abs(y_pred - y_true) < torch.abs(pct_close * y_true)))
    acc = (n_correct.item() * 100.0 / (s*s*batch_size))  # scalar
    return acc



def pytorch_accuracy(y_pred, y_true):
    """
    Computes the accuracy for a batch of predictions

    Args:
        y_pred (torch.Tensor): the logit predictions of the neural network.
        y_true (torch.Tensor): the ground truths.

    Returns:
        The average accuracy of the batch.
    """
    y_pred = y_pred.argmax(1)
    return (y_pred == y_true).float().mean() * 100

def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function , scheduler=scheduler):
    """
    Trains the neural network for one epoch on the train DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer of the neural network
        loss_function: The loss function.

    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the train DataLoader.
    """

    pytorch_network.train(True)

    if scheduler:
        scheduler.step()



    pred = []
    true = []

    with torch.enable_grad():
        loss_sum = 0.
        acc_sum = 0.
        example_count = 0
        for (x, y) in train_loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred, log_var = pytorch_network(x)
            log_var = limit(log_var)

            #modification format pour stat sklearn
            preds = y_pred.flatten().cpu().detach().numpy()
            preds = preds.astype(float)
            target = y.flatten().cpu().detach().numpy()
            target = target.astype(float)

            mask = torch.isnan(y)


            # loss = loss_function(y_pred.squeeze(), y)
            loss = nanmean(negative_log_likelihood(y_pred.squeeze(), log_var.squeeze(), y), mask)

            #loss = custom_mean_square_error(y_pred.squeeze(), y)


            loss.backward()

            optimizer.step()

            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(target[i])



            mse = mean_squared_error(pred,true)
            r2 = r2_score(pred,true)


            # Since the loss and accuracy are averages for the batch, we multiply
            # them by the the number of examples so that we can do the right
            # averages at the end of the epoch.







            loss_sum += float(loss) * len(x)
            acc_sum += float(Regression_accuracy(y_pred, y)) #* len(x)
            example_count += len(x)

    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count
    return avg_loss, avg_acc, mse, r2

def pytorch_test(pytorch_network, loader, loss_function):
    """
    Tests the neural network on a DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to test.
        loader (torch.utils.data.DataLoader): The DataLoader to test on.
        loss_function: The loss function.

    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the DataLoader.
    """


    pred = []
    true = []


    pytorch_network.eval()
    with torch.no_grad():
        loss_sum = 0.
        acc_sum = 0.
        example_count = 0
        for (x, y) in loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)
            y_pred, log_var = pytorch_network(x)

            #modification format pour stat sklearn
            preds = y_pred.flatten().cpu().detach().numpy()
            preds = preds.astype(float)
            target = y.flatten().cpu().detach().numpy()
            target = target.astype(float)

            log_var = limit(log_var)


            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(target[i])

            #print("pred", pred)
            #print("true", true)

            #Calcul statistique pour regression
            mse = mean_squared_error(pred,true)
            r2 = r2_score(pred, true)


  #Calcul de la différence entre prédiction et label pour l'histogramme des erreurs
            diff = (y_pred.flatten() - y.flatten()).cpu().numpy()

            # loss = loss_function(y_pred.squeeze(), y)
            #loss = custom_mean_square_error(y_pred.squeeze(), y)


            mask = torch.isnan(y)
            print('y_pred',y_pred.squeeze().shape)
            print('log_var',log_var.squeeze().shape)
            print('y',y.shape)


            # loss = loss_function(y_pred.squeeze(), y)
            loss = nanmean(negative_log_likelihood(y_pred.squeeze(), log_var.squeeze(), y), mask)

            #loss = custom_mean_square_error(y_pred.squeeze(), y)

            # Since the loss and accuracy are averages for the batch, we multiply
            # them by the the number of examples so that we can do the right
            # averages at the end of the test.
            loss_sum += float(loss) * len(x)
            acc_sum += float(Regression_accuracy(y_pred, y)) #* len(x)

            example_count += len(x)
    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count
    return avg_loss, avg_acc, mse, r2, diff, pred, true




def pytorch_train(pytorch_network):
    """
    This function transfers the neural network to the right device,
    trains it for a certain number of epochs, tests at each epoch on
    the validation set and outputs the results on the test set at the
    end of training.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.

    Example:
        This function displays something like this:

        .. code-block:: python

            Epoch 1/5: loss: 0.5026924496193726, acc: 84.26666259765625, val_loss: 0.17258917854229608, val_acc: 94.75
            Epoch 2/5: loss: 0.13690324830015502, acc: 95.73332977294922, val_loss: 0.14024296019474666, val_acc: 95.68333435058594
            Epoch 3/5: loss: 0.08836929737279813, acc: 97.29582977294922, val_loss: 0.10380942322810491, val_acc: 96.66666412353516
            Epoch 4/5: loss: 0.06714504160980383, acc: 97.91874694824219, val_loss: 0.09626663728555043, val_acc: 97.18333435058594
            Epoch 5/5: loss: 0.05063822727650404, acc: 98.42708587646484, val_loss: 0.10017542181412378, val_acc: 96.95833587646484
            Test:
                Loss: 0.09501855444908142
                Accuracy: 97.12999725341797
    """

     # Early stopping
    last_loss = 100
    patience = 2
    triggertimes = 0


    print(pytorch_network)

    # Transfer weights on GPU if needed.
    pytorch_network.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss() #fonction de perte pour la regression




    #Liste à remplir pour le graphique de fin
    l_train_loss = []
    l_train_acc = []
    l_valid_loss = []
    l_valid_acc = []

    l_train_mse = []
    l_train_r2 = []
    l_valid_mse = []
    l_valid_r2 = []

    for epoch in range(1, num_epochs + 1):
        # Print Learning Rate
        #print('Epoch:', epoch,'LR:', scheduler.get_lr())
        # Training the neural network via backpropagation
        train_loss, train_acc, train_mse, train_r2 = pytorch_train_one_epoch(pytorch_network, optimizer, loss_function)
        scheduler.step()


        # Validation at the end of the epoch
        valid_loss, valid_acc, valid_mse, valid_r2, _, _,_ = pytorch_test(pytorch_network, valid_loader, loss_function)



        print("Epoch {}/{}: loss: {}, acc: {}, val_loss: {}, val_acc: {}".format(
            epoch, num_epochs, train_loss, train_acc, valid_loss, valid_acc
        ))

        # Early stopping
        current_loss = valid_loss
        print('The Current Loss:', current_loss)

        #if current_loss > last_loss:
        #    triggertimes += 1
        #    print('Trigger Times:', triggertimes)

        #    if triggertimes >= patience:
        #        print('Early stopping!\nStart to test process.')
        #        return model

        #else:
        #    print('trigger times: 0')
        #    triggertimes = 0

        #last_loss = current_loss



        l_train_loss.append(train_loss)
        l_train_acc.append(train_acc)
        l_valid_loss.append(valid_loss)
        l_valid_acc.append(valid_acc)

        l_train_mse.append(train_mse)
        l_train_r2.append(train_r2)
        l_valid_mse.append(valid_mse)
        l_valid_r2.append(valid_r2)


    # Test at the end of the training
    test_loss, test_acc, test_mse, test_r2, error, pred, true = pytorch_test(pytorch_network, test_loader, loss_function)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))

    # calculate mse et r2 des batchs de test
    print('MSE: %.3f, RMSE: %.3f, r2: %.3f' % (test_mse, np.sqrt(test_mse), test_r2))

    #Plot histogramme repartition des erreurs
    #num_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4, 1.5]
    plt.hist(error, facecolor='blue', alpha=0.5)

    print('true',(np.array(true).shape))
    print('pred', (np.array(pred).shape))
    id_result = np.stack((np.array(true), np.array(pred)), axis=1)
    print(id_result.shape)
    pd.DataFrame(id_result).to_csv("/home/beland/PycharmProjects/MyProjectVolume/Results/"+name+"id_result.csv")


    #Plot graphique evolution par epoque
    fig, axes = plt.subplots(2, 2)
    plt.tight_layout()

    axes[0,0].set_title('Train accuracy')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].plot(l_train_acc, label='Train')
    axes[0,0].plot(l_valid_acc, label='Validation')
    axes[0,0].legend()

    axes[1,0].set_title('Train loss')
    axes[1,0].set_xlabel('Epochs')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].plot(l_train_loss, label='Train')
    axes[1,0].plot(l_valid_loss, label='Validation')

    axes[0,1].set_title('Train R2')
    axes[0,1].set_xlabel('Epochs')
    axes[0,1].set_ylabel('r2')
    axes[0,1].plot(l_train_r2, label='Train')
    axes[0,1].plot(l_valid_r2, label='Validation')
    axes[0,1].legend()

    axes[1,1].set_title('Train RMSE')
    axes[1,1].set_xlabel('Epochs')
    axes[1,1].set_ylabel('RMSE')
    axes[1,1].plot(np.sqrt(l_train_mse), label='Train')
    axes[1,1].plot(np.sqrt(l_valid_mse), label='Validation')


    # fig, axes = plt.subplots(2, 1)
    # plt.tight_layout()
    #
    # axes[0].set_title('Train R2')
    # axes[0].set_xlabel('Epochs')
    # axes[0].set_ylabel('r2')
    # axes[0].plot(l_train_r2, label='Train')
    # axes[0].plot(l_valid_r2, label='Validation')
    # axes[0].legend()
    #
    # axes[1].set_title('Train RMSE')
    # axes[1].set_xlabel('Epochs')
    # axes[1].set_ylabel('RMSE')
    # axes[1].plot(np.sqrt(l_train_mse), label='Train')
    # axes[1].plot(np.sqrt(l_valid_mse), label='Validation')


    plt.show()
    plt.savefig("/home/beland/PycharmProjects/MyProjectVolume/Results/"+name+"_graph.png")

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    pytorch_train(model)
    PATH = name+".pt"
    torch.save(model, PATH)
