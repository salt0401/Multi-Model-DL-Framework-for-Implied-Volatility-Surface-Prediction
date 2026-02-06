from model import MultiModel, WeightedSumLoss, SimpleLoss
from dataset import DataProcessor

from argparse import ArgumentParser
import configparser

import torch
from torch import optim
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

def printEpoch(epoch, loss):
    epoch = epoch+1
    if epoch % 10**int(math.log10(epoch)) == 0:
        print('Epoch:', epoch, ', loss:', loss)

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_start_date", type=str, default='20140101')
    parser.add_argument("--train_end_date", type=str, default='20201231')
    parser.add_argument("--test_start_date", type=str, default='20210101')
    parser.add_argument("--test_end_date", type=str, default='20211231')
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=20)

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Set if use gpu or not
    # U can check your gpu, and go to pytorch, install the version fits your cuda
    print('Start')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if use_gpu else "cpu")

    # Set the date range of train data and test data
    # When training the model, the train data should not overlap with the test data
        # Or you don't know if the model is good at solving the problem only for the train data 
        # Study for the exam, don't peek at problems of the exam paper
    train_start_date = datetime.strptime(args.train_start_date, '%Y%m%d')
    train_end_date = datetime.strptime(args.train_end_date, '%Y%m%d')
    test_start_date = datetime.strptime(args.test_start_date, '%Y%m%d')
    test_end_date = datetime.strptime(args.test_end_date, '%Y%m%d')
    
    # Preprocess the data and get the train data
    print('Preprocessing...')
    dp = DataProcessor('../dataset/','2009_2023','TWII.csv', 'prs_dataset_no_fat(clean)')
    dp()
    data_gen = dp.Prepare_prs_dataset(train_start_date, train_end_date, args.epochs)
    print('End of data preprocessing.')

    # Set the model, the loss function, the optimizer, the scheduler
    # Optimizer: once get the loss, it will adjust the model parameter by the loss
        # Different types of optimizer makes different adjustment
    # Scheduler: Set the learning rate at each epoch
        # If loss points the direction to adjust, learning rate is the range of adjustment
    print('Model generating...')
    learning_rate = config['model_sett'].getfloat('learning_rate')
    ensemble_num = config['model_sett'].getint('ensemble_num')
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)
    Model = MultiModel(ensemble_num=ensemble_num).to(device)
    loss_function = WeightedSumLoss(device).to(device)
    optimizer = optim.AdamW(Model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(500, args.epochs, 5)], gamma=0.5)

    print('Training...')
    running_train_loss = [] # The train loss computed at each epoch
    running_test_loss = []

    for epoch, train_loader, test_loader, c6_loader in data_gen:
        Model.train()
 
        #to make the longer train data and shorter c6 data run simutaneously
        train_bar = tqdm(train_loader, f"Train epoch {epoch+1}/{args.epochs}") # progress bar
        c6_iterator = iter(c6_loader)
        for tau_train, logm_train, y_train, yATM_train in train_bar:
            try:
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)
            except StopIteration:
                c6_iterator = iter(c6_loader) 
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)
        
            tau_train, logm_train, y_train, yATM_train = tau_train.to(device), logm_train.to(device), y_train.to(device), yATM_train.to(device)
            tau_syn, logm_syn, yATM_syn = tau_syn.to(device), logm_syn.to(device), yATM_syn.to(device)
            output_train, grad_ttm1_train, grad_logm1_train, grad_logm2_train = Model(tau_train, logm_train, yATM_train)
            output_syn, _, _, grad_logm2_syn = Model(tau_syn, logm_syn, yATM_syn)
        
            totalloss = loss_function(output_train, y_train, logm_train, 
                                 grad_ttm1_train, grad_logm1_train, grad_logm2_train, 
                                 output_syn, logm_syn, grad_logm2_syn)

            optimizer.zero_grad()
            totalloss.backward()
            optimizer.step()
            scheduler.step()

            totalloss = totalloss.cpu().item()
            running_train_loss.append(totalloss)
            
        for tau_test, logm_test, y_test, yATM_test in test_loader:
            Model.eval()

            try:
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)
            except StopIteration:
                c6_iterator = iter(c6_loader) 
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)

            tau_test, logm_test, y_test, yATM_test = tau_test.to(device), logm_test.to(device), y_test.to(device), yATM_test.to(device)
            tau_syn, logm_syn, yATM_syn = tau_syn.to(device), logm_syn.to(device), yATM_syn.to(device)
            output_test, grad_ttm1_test, grad_logm1_test, grad_logm2_test = Model(tau_test, logm_test, yATM_test)
            output_syn, _, _, grad_logm2_syn = Model(tau_syn, logm_syn, yATM_syn)

            totalloss = loss_function(output_test, y_test, logm_test, 
                                 grad_ttm1_test, grad_logm1_test, grad_logm2_test, 
                                 output_syn, logm_syn, grad_logm2_syn)
            
            totalloss = totalloss.cpu().item()
            running_test_loss.append(totalloss)
    
    # Evaluate the result
    # If loss is smaller than the smallest loss recorded, consider the model now is the best trained model
    print('Evaluating result...')
    Model.eval()
    test_loss = 0
    plot_path = config['save_path']['plot_path']
    model_path = config['save_path']['model_path']
    best_loss = config['model_sett'].getfloat('best_loss')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    if totalloss < best_loss:
        config['model_sett']['best_loss'] = str(totalloss)
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        torch.save(Model.state_dict(), model_path)
    
    # Make a plot to check how the training
    # y-axis is the loss at each epoch
    # Under best situation, the loss should be decreasing with no big fluctuation
        # And the train loss should not has too much diff from test loss
    train_start_date = train_start_date.strftime("%Y%m%d")
    train_end_date = train_end_date.strftime("%Y%m%d")
    test_start_date = test_start_date.strftime("%Y%m%d")
    test_end_date = test_end_date.strftime("%Y%m%d")
    plt.plot(running_train_loss, label='Training Loss')
    plt.plot(running_test_loss, label='Testing Loss')
    plt.xlabel('running')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/train{train_start_date}to{train_end_date}_test{test_start_date}to{test_end_date}_{epoch}epoch.png')
    print('Finished Training')

if __name__ == '__main__':
    main()