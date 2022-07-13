import random
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import pandas as pd
from torch import nn
import os
from torch.utils.data import DataLoader, TensorDataset
import itertools
import time
import copy
from LSTM_glove import SentimentLSTM

# create Tensor datasets
train_data, train_labels = np.load('old_data/files_glove/Train_data.npy'), np.load('old_data/files_glove/Train_labels.npy')
test_data, test_labels = np.load('old_data/files_glove/Test_data.npy'), np.load('old_data/files_glove/Test_labels.npy')
train_data = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))


def train_model(model, device,batch_size, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    trigger_times = 0
    loss_dict = {'train': [], 'test': []}
    acc_dict = {'train': [], 'test': []}
    patience = int(num_epochs//6)+1
    best_acc = 0
    predictions = {'train': [], 'test': []}
    best_predictions = {}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        h = model.init_hidden(batch_size)
        # Each epoch has a training and validation phase
        for phase in ['train',  'test']:
            predicted_list = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            correct = 0
            total = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if len(labels) != batch_size:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                print(f'input size is :{inputs.size()}')
                # zero the parameter gradients
                optimizer.zero_grad()
                # h = tuple([each.data for each in h])
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs, h = model(inputs.long())
                    _, preds = torch.max(outputs, 1, keepdim=True)  # check the labels
                    loss = criterion(outputs.squeeze(), labels.long())  # compute loss
                    predicted_list.append(preds)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # backward
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()  # Optimization step - update weights

                # statistics
                running_loss += loss.item() * inputs.size(0)
                correct += (preds == labels.reshape(-1,1)).sum()
                total += labels.size(0)

            if phase == 'train':
                scheduler.step()

            predictions[phase] = copy.deepcopy(predicted_list)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(correct) / total


            print('{} Loss: {:.4f} Accuracy: {:.4f} '.format(phase, epoch_loss, epoch_acc))

            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            # early stopping
            if phase == 'test':
                if epoch_acc > best_acc:
                    trigger_times = 0
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_predictions = copy.deepcopy(predictions)
                else:
                    if epoch > 5:
                        trigger_times += 1
                        print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    model.load_state_dict(best_model_wts)

                    print('Early stopping!\nStart to test process.')
                    return model, loss_dict, acc_dict, best_acc, best_predictions
    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_dict, acc_dict, best_acc, best_predictions


def runner(model, lr=0.01, batch_size=32, num_epochs=20):
    # Load datasets
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    dataloaders = {'train': train_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_data), 'test': len(test_data)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return train_model(model, device,batch_size, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs)

## hyperparameter optimization
HPO = {
    'batch_size': [64, 32, 128],
    'lr': [0.005, 0.1, 0.001, 0.01, 0.05],
    'epochs': [30, 60, 100],
    'n_layers': [1, 2, 3, 4, 5],
    'n_hidden_units': [32, 64, 128],
    'drop_prob': [0.5, 0.7, 0.1, 0.2, 0.3, 0.4]
}

## hyperparameter optimization
# HPO_BEST = {
#     'batch_size' : [32],
#     'lr' : [0.005],
#     'epochs': [30],
#     'n_layers' : [1],
#     'n_hidden_units': [32],
#     'drop_prob': [0.1],
#     'n_emb': [100]
#
# }

# HPO  = HPO_BEST
## create permutations of all possible hyperparameters
All_params = list(itertools.product(*(HPO[param] for param in HPO.keys())))
random.shuffle(All_params)
output_size = 3
path = 'old_data/transformations' ## keep for local run
# path = 'HW2/models/LSTM_glove_new'

############### Simple LSTM glove loop###############
####################################################################################################################
for i in range(len(All_params)):
    curr_params = All_params[i]
    batch_size, lr, epochs, n_layers, n_hidden, drop_prob = curr_params[0], curr_params[1], \
                                                            curr_params[2], curr_params[3], \
                                                            curr_params[4], curr_params[5]
    curr_params = f'lr={lr}_b={batch_size}_t={epochs}_n_layers={n_layers}_n_hid={n_hidden}_p={drop_prob}'
    print('Starting parameters:')
    print(curr_params)
    all_files = os.listdir(path + '/loss')
    ## check if current params already exist in folder
    continue_flag = False
    if len(all_files) > 0:
        all_files_split = [x.split('_best_acc')[0] for x in all_files]
        if curr_params in all_files_split:
            print('skipping params: ')
            print(curr_params)
            continue_flag = True
    if continue_flag:
        continue

    weights_puncs = np.load('old_data/files_glove/weight_mat.npy')
    weights_puncs = torch.from_numpy(weights_puncs)
    model = SentimentLSTM(weights_puncs, output_size, n_hidden, n_layers, drop_prob)
    trained_model, loss_dict, acc_dict, best_acc, best_predictions = runner(model, lr, batch_size, epochs)
    results = f"_best_acc={round(best_acc, 4)}"
    file_name = curr_params + results
    torch.save(trained_model.state_dict(), os.path.join(path, 'pkl/' + file_name + '.pkl'))
    loss_df = pd.DataFrame(loss_dict)
    acc_df = pd.DataFrame(acc_dict)
    loss_df.to_csv(os.path.join(path, 'loss/' + file_name))
    acc_df.to_csv(os.path.join(path, 'acc/' + file_name))





#################################################################################
## Simple LSTM loop -  first experiments - without GLOVE
# for i in range(len(All_params)):
#     curr_params = All_params[i]
#     batch_size, lr, epochs, n_layers, n_hidden, drop_prob, n_emb = curr_params[0], curr_params[1], \
#                                                             curr_params[2], curr_params[3], \
#                                                             curr_params[4], curr_params[5], curr_params[6]
#     curr_params = f'lr={lr}_b={batch_size}_t={epochs}_n_layers={n_layers}_n_hid={n_hidden}_p={drop_prob}_emb={n_emb}'
#     print('Starting parameters:')
#     print(curr_params)
#     all_files = os.listdir(path + '/loss')
#     continue_flag = False
#     if len(all_files) > 0:
#         all_files_split = [x.split('_best_acc')[0] for x in all_files]
#         if curr_params in all_files_split:
#             print('skipping params: ')
#             print(curr_params)
#             continue_flag = True
#     if continue_flag:
#         continue
#
#     model = SentimentLSTM(vocab_size, output_size, n_emb, n_hidden, n_layers, drop_prob)
#     trained_model, loss_dict, acc_dict, best_acc, best_predictions = runner(model, lr, batch_size, epochs)
#     results = f'_best_acc={round(best_acc, 4)}'
#     file_name = curr_params + results
#     torch.save(trained_model.state_dict(), os.path.join(path, 'pkl/' + file_name + '.pkl'))
#     loss_df = pd.DataFrame(loss_dict)
#     acc_df = pd.DataFrame(acc_dict)
#     # best_preds_df = pd.DataFrame(best_predictions)
#     loss_df.to_csv(os.path.join(path, 'loss/' + file_name))
#     acc_df.to_csv(os.path.join(path, 'acc/' + file_name))
#     # best_preds_df.csv(os.path.join(path, 'preds/' + file_name))


