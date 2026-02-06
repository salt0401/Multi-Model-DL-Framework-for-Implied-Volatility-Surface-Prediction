import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import time
from ast import literal_eval
from itertools import product
'''
row = {
    'strike_price':3600.0,
    'put/call':'put',
    'option_price':4.0,
    'tau':0.06349206349206349,
    'underlying':4698.292969,
    'logm':-0.25875015849674393
}
'''

def BS(S_beta, K_beta, logm, tau, implied_vol, indic='call'):
        d1 = -logm/np.sqrt(tau)/implied_vol + np.sqrt(tau)*implied_vol/2
        d2 = -logm/np.sqrt(tau)/implied_vol - np.sqrt(tau)*implied_vol/2
        if indic == 'call':
            return S_beta*norm.cdf(d1)-K_beta*norm.cdf(d2)
        elif indic == 'put':
            return K_beta*norm.cdf(-d2)-S_beta*norm.cdf(-d1)

def implied_vol(row):
    
    def error(sigma):
        return (BS(row['underlying_beta'], row['strike_price_beta'], row['logm'], row['tau'], sigma, row['put/call']) - row['option_price'])**2
    
    return minimize(error, x0=[0.2], bounds=[(0.0001, 1)])['x'][0]


#df = pd.read_csv('../dataset/2009_2023_impliVol.csv')
#df = pd.read_csv('../dataset/2009_2023_intDivConcatnated.csv')
#print(df.head())
#df_min = df[df['implied_vol'] == 0.001].drop('implied_vol', axis=1)
#df_part = df.iloc[1:10000, :]
'''
st = time.time()
df_part['implied_vol'] = df_part.apply(implied_vol, axis=1)
et = time.time()
print(et-st)
'''
#df_min = df[df['implied_vol'] == 0.001]
#print(df_min.shape)
df = pd.read_csv('../dataset/prs_dataset_no_fat(clean).csv', index_col=0)
df['logm'] = np.log(df['strike_price']/df['S']) - df['tau']*(df['d'] - df['r'])
df['total_var'] = np.square(df['impl_volatility'])*df['tau']
print(df.head())
df.to_csv('../dataset/prs_dataset_no_fat(clean).csv')

taus = df['tau'].unique()
min_logm = df['logm'].min()
max_logm = df['logm'].max()

logm_seq = [min_logm * 6, min_logm * 5, min_logm * 4, max_logm * 4, max_logm * 5, max_logm * 6]


# Create a new DataFrame (tibble) with the unique values
product_result_df = pd.DataFrame([(tau, logm) for tau in taus for logm in logm_seq],
                                 columns=['tau', 'logm'])
print(product_result_df.head())
product_result_df.to_csv('../dataset/syn_data_c6.csv')