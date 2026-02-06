from model import MultiModel, WeightedSumLoss, SimpleLoss
from dataset import DataProcessor
from argparse import ArgumentParser

from datetime import datetime

import configparser
import torch

from scipy.stats import norm
import numpy as np
import pandas as pd

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)

def experiment(start_date, end_date, Model, tau, logm, y, y_atm, test_df):
    y_hat = Model(tau, logm, y_atm).cpu().detach().numpy().flatten()
    test_df['impli_vol_pred'] = np.sqrt(y_hat/test_df['tau'])
    print(test_df.head())

def main():
    parser = ArgumentParser()
    parser.add_argument("--start_date", type=str, default='20210101')
    parser.add_argument("--end_date", type=str, default='20211231')
    parser.add_argument("--on_gpu", action='store_true')
    
    args = parser.parse_args()
    start_date = datetime.strptime(args.start_date, '%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%Y%m%d')

    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Start')
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(42)

    print('Preprocessing...')
    dp = DataProcessor('../dataset/','2009_2023','TWII.csv', 'prs_dataset_no_fat(clean)')
    dp()
    tau, logm, y, y_atm, test_df = dp.Prepare_test_data(start_date, end_date)
    print('End of data preprocessing.')

    print('Model loading...')
    model_path = config['save_path']['model_path']
    Model = MultiModel().to(device)
    Model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    loss_function = SimpleLoss().to(device)

    print('Experimenting...')
    experiment(start_date, end_date, Model, tau, logm, y, y_atm, test_df)

if __name__ == '__main__':
    main()