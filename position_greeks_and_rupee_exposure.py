from py5paisa import FivePaisaClient
import json
import pandas as pd
from py5paisa.order import Order, OrderType, Exchange
cred={
    "APP_NAME":"",
    "APP_SOURCE":"",
    "USER_ID":"",
    "PASSWORD":"",
    "USER_KEY":"",
    "ENCRYPTION_KEY":""
    }

client = FivePaisaClient(email="", passwd="", dob="",cred=cred)
client.login()
import requests
import glob
import os
import warnings
import time
from scipy.stats import norm
from math import sqrt, log
import numpy as np
from nsepython import *

warnings.simplefilter("ignore")

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

underlying = 'NIFTY'


folder_path = 'your/folder/path/here/'
sample_file = 'sample.csv'
list_of_files = glob.glob(folder_path + sample_file)
latest_file = max(list_of_files, key=os.path.getctime)
options_df = pd.read_csv(latest_file)
positions = client.positions()
positions_df = pd.DataFrame(positions)
open_positions = positions_df[positions_df['NetQty'] != 0]
open_positions = open_positions.reset_index()
open_positions = open_positions.drop('index', 1)
open_positions_v2 = open_positions[['ScripCode', 'ScripName', 'NetQty', 'Multiplier', 'LTP']]
open_positions_v2['CurrentValue'] = open_positions_v2['NetQty'] * open_positions_v2['Multiplier'] * open_positions_v2['LTP'] * 1.00
open_positions_v2 = open_positions_v2[['ScripCode', 'ScripName', 'NetQty', 'Multiplier', 'LTP', 'CurrentValue']]
open_positions_v2 = open_positions_v2.rename(columns={'ScripCode': 'Scripcode'})

# Scrip Master file downloaded from 5Paisa documentation page
path = 'path/to/5Paisa/ScripMaster/File.csv'
scrips = pd.read_csv(path)
temp = open_positions_v2[['Scripcode']]
temp = temp.merge(scrips, on=['Scripcode'], how='inner')
temp = temp[['Scripcode', 'Expiry', 'StrikeRate', 'CpType', 'Root']]
nifty_temp = temp[temp['Root'] == underlying]
nifty = open_positions_v2.merge(nifty_temp, on=['Scripcode'], how='inner')
nifty = nifty.rename(columns={
    'Root': 'underlying',
    'CpType': 'option_type',
    'StrikeRate': 'strike',
    'Expiry': 'expiry',
    'CurrentValue': 'current_value',
    'LTP': 'ltp',
    'Multiplier': 'multiplier',
    'NetQty': 'net_qty',
    'ScripName': 'scrip_name',
    'Scripcode': 'scrip_code'
})
nifty['expiry'] = nifty['expiry'].str.replace('14:30:00','15:30:00')
nifty['expiry'] = pd.to_datetime(nifty['expiry'])
options_df['expiry'] = pd.to_datetime(options_df['expiry'])
nifty = pd.merge(nifty, options_df,  how='inner', left_on=['expiry', 'strike'], right_on = ['expiry', 'strike'])
nifty = nifty[[
    'scrip_code', 'scrip_name', 'underlying', 'expiry', 'strike', 'option_type',
    'current_value', 'multiplier', 'net_qty', 'ltp', 'rf_rate', 'call_spot_price', 'forward',
    'otm_option_type', 'timestamp', 'implied_volatility', 'd1', 'd2', 'days_to_expiry', 
    'f_minus_k', 'delta', 'gamma', 'theta', 'vega', 'rho', 'vanna', 'charm', 'volga'
]]

c1 = ((nifty['option_type'] == 'PE') & (nifty['f_minus_k'] >= 0))
c2 = ((nifty['option_type'] == 'CE') & (nifty['f_minus_k'] <= 0))
itm_or_otm = ['otm', 'otm']
default = 'itm'
nifty['itm_or_otm'] = np.select([c1, c2], itm_or_otm, default=default)

c1 = ((nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
c2 = ((nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
delta = [nifty['delta'] - 1, 1 + nifty['delta']]
default = nifty['delta']
nifty['delta'] = np.select([c1, c2], delta, default=default)

r = nifty['rf_rate']
t = (nifty['days_to_expiry']) / 365.0
d1 = nifty['d1']
d2 = nifty['d2']
F = nifty['forward']
σ = nifty['implied_volatility']
K = nifty['strike']
N_prime = norm.pdf
N = norm.cdf

c1 = ((nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
c2 = ((nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
charm = [round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4), round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4)]
default = nifty['charm']
nifty['charm'] = np.select([c1, c2], delta, default=default)

nifty['delta_pos'] = nifty['delta'] * nifty['net_qty'] * nifty['multiplier']
nifty['gamma_pos'] = nifty['gamma'] * nifty['net_qty'] * nifty['multiplier']
nifty['theta_pos'] = nifty['theta'] * nifty['net_qty'] * nifty['multiplier']
nifty['vega_pos'] = nifty['vega'] * nifty['net_qty'] * nifty['multiplier']

typical_move = nifty['forward'].iloc[-1] * 12.62 * 0.01 / 16
typical_iv_change = -0.1
agg_delta = nifty['delta_pos'].sum()
agg_gamma = nifty['gamma_pos'].sum()
agg_theta = nifty['theta_pos'].sum()
agg_vega = nifty['vega_pos'].sum()

ul_price_exp_up = typical_move * agg_delta + typical_move * typical_move * agg_gamma / 2
ul_price_exp_down = - typical_move * agg_delta + typical_move * typical_move * agg_gamma / 2

time_exp = agg_theta

nifty['typical_iv_change'] = typical_iv_change * nifty['implied_volatility']
nifty['vol_exp'] = nifty['typical_iv_change'] * nifty['vega_pos'] * 100
vol_exp = nifty['vol_exp'].sum()

print("Exposures | Assuming no change in the greeks")
print("\n")
print("Exposure to underlying price movement (UP): ", round(ul_price_exp_up))
print("Exposure to underlying price movement (DOWN): ", round(ul_price_exp_down))
print("Exposure to time: ", round(time_exp))
print("Exposure to volatility: ", round(vol_exp))

print("\n")
print("Exposures | Change in underlying price")
moves = [
    - 2 * typical_move, - typical_move, typical_move, 2 * typical_move
]
for move in moves:
    nifty['new_forward'] = nifty['forward'] + move
    nifty['new_d1'] = (np.log(nifty['new_forward'] / nifty['strike']) + (0.5 * nifty['implied_volatility'] ** 2 ) * (nifty['days_to_expiry'] / 365.0)) / (nifty['implied_volatility'] * np.sqrt((nifty['days_to_expiry'] / 365.0)))
    nifty['new_d2'] = nifty['new_d1'] - nifty['implied_volatility'] * np.sqrt((nifty['days_to_expiry'] / 365.0))
    nifty['new_d1'] = pd.to_numeric(nifty['new_d1'], errors='coerce')
    nifty['new_d2'] = pd.to_numeric(nifty['new_d2'], errors='coerce')
    
    r = nifty['rf_rate']
    t = (nifty['days_to_expiry']) / 365.0
    d1 = nifty['new_d1']
    d2 = nifty['new_d2']
    F = nifty['new_forward']
    σ = nifty['implied_volatility']
    K = nifty['strike']

    c1 = (nifty['days_to_expiry']) == 0
    c2 = (((nifty['days_to_expiry']) != 0) & (nifty['option_type'] == 'CE'))
    c3 = (((nifty['days_to_expiry']) != 0) & (nifty['option_type'] == 'PE'))
    c4 = (nifty['days_to_expiry']) != 0
    c5 = (((nifty['days_to_expiry']) == 0) & (nifty['option_type'] == 'CE') & (nifty['f_minus_k'] > 0))
    c6 = (((nifty['days_to_expiry']) == 0) & (nifty['option_type'] == 'PE') & (nifty['f_minus_k'] < 0))

    delta = [0, np.exp(-r*t)*N(d1), np.exp(-r*t)*(-N(-d1)), 1, 1]
    gamma = [0, round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10)]
    theta = [0, round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4)]
    vega = [0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4)]
    rho = [0, t*K*np.exp(-r*t)*N(d2), -t*K*np.exp(-r*t)*N(-d2)]
    vanna = [0, round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10)]
    charm = [0, round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4), round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4)]
    volga = [0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4)]
    default = 0

    nifty['new_delta'] = np.select([c1, c2, c3, c5, c6], delta, default=default)
    nifty['new_gamma'] = np.select([c1, c4], gamma, default=default)
    nifty['new_theta'] = np.select([c1, c4], theta, default=default)
    nifty['new_vega'] = np.select([c1, c4], vega, default=default)
    nifty['new_rho'] = np.select([c1, c2, c3], rho, default=default)
    nifty['new_vanna'] = np.select([c1, c4], vanna, default=default)
    nifty['new_charm'] = np.select([c1, c2, c3], charm, default=default)
    nifty['new_volga'] = np.select([c1, c4], volga, default=default)
    
    nifty['new_delta_pos'] = nifty['new_delta'] * nifty['net_qty'] * nifty['multiplier']
    nifty['new_gamma_pos'] = nifty['new_gamma'] * nifty['net_qty'] * nifty['multiplier']
    
    c1 = (((nifty['days_to_expiry']) == 0) & (nifty['option_type'] == 'PE'))
    c2 = (((nifty['days_to_expiry']) == 0) & (nifty['option_type'] == 'CE'))
    c3 = (((nifty['days_to_expiry']) != 0) & (nifty['option_type'] == 'PE'))
    c4 = (((nifty['days_to_expiry']) != 0) & (nifty['option_type'] == 'CE'))
    default = 0
    price = [K-F, F-K, (-F*N(-d1)+N(-d2)*K)*np.exp(-r*t), (F*N(d1)-N(d2)*K)*np.exp(-r*t)]
    nifty['new_price'] = np.select([c1, c2, c3, c4], price, default=default)
    nifty['new_value_pos'] = nifty['new_price'] * nifty['net_qty'] * nifty['multiplier']
    
    print("\n")
    print("Underlying price change:", move)
    print("Delta:", round(nifty['delta_pos'].sum()), round(nifty['new_delta_pos'].sum()))
    print("Gamma:", round(nifty['gamma_pos'].sum(), 4), round(nifty['new_gamma_pos'].sum(), 4))
    print("PnL:", round(nifty['new_value_pos'].sum() - nifty['current_value'].sum()))
    
print("\n")
print("Exposures | Change in time")
dtes = [1, 2, 5, 10]
for dte in dtes:
    nifty['new_dte'] = nifty['days_to_expiry'] - dte
    nifty['new_d1'] = (np.log(nifty['forward'] / nifty['strike']) + (0.5 * nifty['implied_volatility'] ** 2 ) * (nifty['new_dte'] / 365.0)) / (nifty['implied_volatility'] * np.sqrt((nifty['new_dte'] / 365.0)))
    nifty['new_d2'] = nifty['new_d1'] - nifty['implied_volatility'] * np.sqrt((nifty['new_dte'] / 365.0))
    nifty['new_d1'] = pd.to_numeric(nifty['new_d1'], errors='coerce')
    nifty['new_d2'] = pd.to_numeric(nifty['new_d2'], errors='coerce')
    
    r = nifty['rf_rate']
    t = (nifty['new_dte']) / 365.0
    d1 = nifty['new_d1']
    d2 = nifty['new_d2']
    F = nifty['forward']
    σ = nifty['implied_volatility']
    K = nifty['strike']
    
    c1 = (((nifty['new_dte']) <= 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
    c2 = (((nifty['new_dte']) <= 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'otm'))
    c3 = (((nifty['new_dte']) <= 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
    c4 = (((nifty['new_dte']) <= 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'otm'))
    c5 = (((nifty['new_dte']) > 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
    c6 = (((nifty['new_dte']) > 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'otm'))
    c7 = (((nifty['new_dte']) > 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
    c8 = (((nifty['new_dte']) > 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'otm'))

    delta = [-1, 0, 1, 0, np.exp(-r*t)*(-N(-d1)), np.exp(-r*t)*(-N(-d1)), np.exp(-r*t)*N(d1), np.exp(-r*t)*N(d1)]
    gamma = [0, 0, 0, 0, round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10)]
    theta = [0, 0, 0, 0, round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4)]
    vega = [0, 0, 0, 0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4)]
    rho = [0, 0, 0, 0, -t*K*np.exp(-r*t)*N(-d2), -t*K*np.exp(-r*t)*N(-d2), t*K*np.exp(-r*t)*N(d2), t*K*np.exp(-r*t)*N(d2)]
    vanna = [0, 0, 0, 0, round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10)]
    charm = [0, 0, 0, 0, round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4), round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4), round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4), round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4)]
    volga = [0, 0, 0, 0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4)]
    price = [K - F, 0, F - K, 0, (-F*N(-d1)+N(-d2)*K)*np.exp(-r*t), (-F*N(-d1)+N(-d2)*K)*np.exp(-r*t), (F*N(d1)-N(d2)*K)*np.exp(-r*t), (F*N(d1)-N(d2)*K)*np.exp(-r*t)]
    default = 0

    nifty['new_delta'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], delta, default=default)
    nifty['new_gamma'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], gamma, default=default)
    nifty['new_theta'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], theta, default=default)
    nifty['new_vega'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], vega, default=default)
    nifty['new_rho'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], rho, default=default)
    nifty['new_vanna'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], vanna, default=default)
    nifty['new_charm'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], charm, default=default)
    nifty['new_volga'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], volga, default=default)
    nifty['new_price'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], price, default=default)
    
    nifty['new_theta_pos'] = nifty['new_theta'] * nifty['net_qty'] * nifty['multiplier']
    nifty['new_value_pos'] = nifty['new_price'] * nifty['net_qty'] * nifty['multiplier']
    
    print("\n")
    print("Change in days to expiry:", dte)
    print("Theta:", round(nifty['theta_pos'].sum()), round(nifty['new_theta_pos'].sum()))
    print("PnL:", round(nifty['new_value_pos'].sum() - nifty['current_value'].sum()))
    
print("\n")
print("Exposures | Change in iVol")
iv_changes = [-0.1, -0.05, 0.05, 0.1]
for iv_change in iv_changes:
    nifty['new_iv'] = (1 + iv_change) * nifty['implied_volatility']
    nifty['new_d1'] = (np.log(nifty['forward'] / nifty['strike']) + (0.5 * nifty['new_iv'] ** 2 ) * (nifty['days_to_expiry'] / 365.0)) / (nifty['new_iv'] * np.sqrt((nifty['days_to_expiry'] / 365.0)))
    nifty['new_d2'] = nifty['new_d1'] - nifty['new_iv'] * np.sqrt((nifty['days_to_expiry'] / 365.0))
    nifty['new_d1'] = pd.to_numeric(nifty['new_d1'], errors='coerce')
    nifty['new_d2'] = pd.to_numeric(nifty['new_d2'], errors='coerce')
    
    r = nifty['rf_rate']
    t = (nifty['days_to_expiry']) / 365.0
    d1 = nifty['new_d1']
    d2 = nifty['new_d2']
    F = nifty['forward']
    σ = nifty['new_iv']
    K = nifty['strike']
    
    c1 = (((nifty['days_to_expiry']) <= 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
    c2 = (((nifty['days_to_expiry']) <= 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'otm'))
    c3 = (((nifty['days_to_expiry']) <= 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
    c4 = (((nifty['days_to_expiry']) <= 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'otm'))
    c5 = (((nifty['days_to_expiry']) > 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'itm'))
    c6 = (((nifty['days_to_expiry']) > 0) & (nifty['option_type'] == 'PE') & (nifty['itm_or_otm'] == 'otm'))
    c7 = (((nifty['days_to_expiry']) > 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'itm'))
    c8 = (((nifty['days_to_expiry']) > 0) & (nifty['option_type'] == 'CE') & (nifty['itm_or_otm'] == 'otm'))

    delta = [-1, 0, 1, 0, np.exp(-r*t)*(-N(-d1)), np.exp(-r*t)*(-N(-d1)), np.exp(-r*t)*N(d1), np.exp(-r*t)*N(d1)]
    gamma = [0, 0, 0, 0, round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10), round(np.exp(-r*t)*N_prime(d1)/(F*σ*np.sqrt(t)), 10)]
    theta = [0, 0, 0, 0, round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4), round(- F*σ*N_prime(d1)/(2*np.sqrt(t))/365.0000, 4)]
    vega = [0, 0, 0, 0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)/100.0000, 4)]
    rho = [0, 0, 0, 0, -t*K*np.exp(-r*t)*N(-d2), -t*K*np.exp(-r*t)*N(-d2), t*K*np.exp(-r*t)*N(d2), t*K*np.exp(-r*t)*N(d2)]
    vanna = [0, 0, 0, 0, round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10), round(-np.exp(-r*t)*d2*N_prime(d1)/σ/100.0000, 10)]
    charm = [0, 0, 0, 0, round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4), round(np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))-(-r*N(-d1))), 4), round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4), round(-np.exp(-r*t)*(N_prime(d1)*(-d2/(2*t))+(-r*N(d1))), 4)]
    volga = [0, 0, 0, 0, round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4), round(F*np.exp(-r*t)*N_prime(d1)*np.sqrt(t)*d1*d2/σ / 100.0000, 4)]
    price = [K - F, 0, F - K, 0, (-F*N(-d1)+N(-d2)*K)*np.exp(-r*t), (-F*N(-d1)+N(-d2)*K)*np.exp(-r*t), (F*N(d1)-N(d2)*K)*np.exp(-r*t), (F*N(d1)-N(d2)*K)*np.exp(-r*t)]
    default = 0

    nifty['new_delta'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], delta, default=default)
    nifty['new_gamma'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], gamma, default=default)
    nifty['new_theta'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], theta, default=default)
    nifty['new_vega'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], vega, default=default)
    nifty['new_rho'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], rho, default=default)
    nifty['new_vanna'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], vanna, default=default)
    nifty['new_charm'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], charm, default=default)
    nifty['new_volga'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], volga, default=default)
    nifty['new_price'] = np.select([c1, c2, c3, c4, c5, c6, c7, c8], price, default=default)
    
    nifty['new_vega_pos'] = nifty['new_vega'] * nifty['net_qty'] * nifty['multiplier']
    nifty['new_value_pos'] = nifty['new_price'] * nifty['net_qty'] * nifty['multiplier']
    
    print("\n")
    print("Change in implied volatility in %:", iv_change * 100, "%")
    print("Vega:", round(nifty['vega_pos'].sum()), round(nifty['new_vega_pos'].sum()))
    print("PnL:", round(nifty['new_value_pos'].sum() - nifty['current_value'].sum()))