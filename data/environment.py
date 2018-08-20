# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 05:41:28 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time

eps=10e-8

def fill_zeros(x):
    return '0'*(6-len(x))+x

class Environment:
    def __init__(self,start_date,end_date,codes,features,window_length,market):#,test_data,assets_list,M,L,N,start_date,end_date

        #preprocess parameters
        self.cost=0.0025

        #read all data
        data=pd.read_csv(r'./data/'+market+'.csv',index_col=0,parse_dates=True,dtype=object)
        data["code"]=data["code"].astype(str)
        if market=='China':
            data["code"]=data["code"].apply(fill_zeros)

        data=data.loc[data["code"].isin(codes)]
        data[features]=data[features].astype(float)

        # 生成有效时间
        start_date = [date for date in data.index if date > pd.to_datetime(start_date)][0]
        end_date = [date for date in data.index if date < pd.to_datetime(end_date)][-1]
        data=data[start_date.strftime("%Y-%m-%d"):end_date.strftime("%Y-%m-%d")]
        #TO DO:REFINE YOUR DATA

        #Initialize parameters
        self.M=len(codes)+1
        self.N=len(features)
        self.L=window_length

        #为每一个资产生成数据
        asset_dict=dict()#每一个资产的数据
        datee=data.index.unique()
        self.date_len=len(datee)
        for asset in codes:
            asset_data=data[data["code"]==asset].reindex(datee).sort_index()#加入时间的并集，会产生缺失值pd.to_datetime(self.date_list)
            asset_data['close']=asset_data['close'].fillna(method='pad')
            base_price = asset_data.ix[end_date, 'close']
            asset_dict[str(asset)]= asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            if 'high' in features:
                asset_dict[str(asset)]['high'] = asset_dict[str(asset)]['high'] / base_price

            if 'low' in features:
                asset_dict[str(asset)]['low']=asset_dict[str(asset)]['low']/base_price

            if 'open' in features:
                asset_dict[str(asset)]['open']=asset_dict[str(asset)]['open']/base_price

            if 'PE' in features:
                asset_data['PE']=asset_data['PE'].fillna(method='pad')
                base_PE=asset_data.ix[end_date,'PE']
                asset_dict[str(asset)]['PE'] = asset_dict[str(asset)]['PE'] / base_PE

            if 'PB' in features:
                asset_data['PB'] = asset_data['PB'].fillna(method='pad')
                base_PB=asset_data.ix[end_date,'PB']
                asset_dict[str(asset)]['PB'] = asset_dict[str(asset)]['PB'] / base_PB

            if 'TR'in features:
                asset_data['TR'] = asset_data['TR'].fillna(method='pad')
                base_TR=asset_data.ix[end_date,'TR']

            if 'TV1' in features:
                base_TV1=asset_data.ix[end_date,'TV1']
                asset_dict[str(asset)]['TV1'] = asset_dict[str(asset)]['TV1'] / base_TV1

            if 'TV2' in features:
                base_TV2=asset_data.ix[end_date,'TV2']
                asset_dict[str(asset)]['TV2'] = asset_dict[str(asset)]['TV2'] / base_TV2

            if 'TR' in features:
                base_TR=asset_data.ix[end_date,'TR']
                asset_dict[str(asset)]['TR'] = asset_dict[str(asset)]['TR'] / base_TR

            asset_data=asset_data.fillna(method='bfill',axis=1)
            asset_data=asset_data.fillna(method='ffill',axis=1)#根据收盘价填充其他值
            #***********************open as preclose*******************#
            #asset_data=asset_data.dropna(axis=0,how='any')
            asset_data=asset_data.drop(columns=['code'])
            asset_dict[str(asset)]=asset_data


        #开始生成tensor
        self.states=[]
        self.price_history=[]
        print("*-------------Now Begin To Generate Tensor---------------*")
        t =self.L+1
        while t<self.date_len:
            V_close = np.ones(self.L)
            if 'high' in features:
                V_high=np.ones(self.L)
            if 'open' in features:
                V_open=np.ones(self.L)
            if 'low' in features:
                V_low=np.ones(self.L)
            if 'TV1' in features:
                V_TV1=np.ones(self.L)
            if 'TV2' in features:
                V_TV2=np.ones(self.L)
            if 'DA' in features:
                V_DA=np.ones(self.L)
            if 'TR' in features:
                V_TR=np.ones(self.L)
            if 'PE' in features:
                V_PE=np.ones(self.L)
            if 'PB' in features:
                V_PB=np.ones(self.L)

            y=np.ones(1)
            for asset in codes:
                asset_data=asset_dict[str(asset)]
                V_close = np.vstack((V_close, asset_data.ix[t - self.L - 1:t - 1, 'close']))
                if 'high' in features:
                    V_high=np.vstack((V_high,asset_data.ix[t-self.L-1:t-1,'high']))
                if 'low' in features:
                    V_low=np.vstack((V_low,asset_data.ix[t-self.L-1:t-1,'low']))
                if 'open' in features:
                    V_open=np.vstack((V_open,asset_data.ix[t-self.L-1:t-1,'open']))
                if 'TV1' in features:
                    V_TV1 = np.vstack((V_TV1, asset_data.ix[t - self.L - 1:t - 1, 'TV1']))
                if 'TV2' in features:
                    V_TV2 = np.vstack((V_TV2, asset_data.ix[t - self.L - 1:t - 1, 'TV2']))
                if 'DA' in features:
                    V_DA = np.vstack((V_DA, asset_data.ix[t - self.L - 1:t - 1, 'DA']))
                if 'TR' in features:
                    V_TR=np.vstack((V_TR,asset_data.ix[t-self.L-1:t-1,'TR']))
                if 'PE' in features:
                    V_PE=np.vstack((V_PE,asset_data.ix[t-self.L-1:t-1,'PE']))
                if 'PB' in features:
                    V_PB=np.vstack((V_PB,asset_data.ix[t-self.L-1:t-1,'PB']))
                y=np.vstack((y,asset_data.ix[t,'close']/asset_data.ix[t-1,'close']))
            state = V_close
            if 'high' in features:
                state = np.stack((state,V_high), axis=2)
            if 'low' in features:
                state = np.stack((state,V_low), axis=2)
            if 'open' in features:
                state = np.stack((state,V_open), axis=2)
            if 'TV1' in features:
                state = np.stack((state,V_TV1), axis=2)
            if 'TV2' in features:
                state = np.stack((state,V_TV2), axis=2)
            if 'DA' in features:
                state = np.stack((state,V_DA), axis=2)
            if 'TR' in features:
                state = np.stack((state,V_TR), axis=2)
            if 'PE' in features:
                state = np.stack((state,V_PE), axis=2)
            if 'PB' in features:
                state = np.stack((state,V_PB), axis=2)
            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)
            self.price_history.append(y)
            t=t+1
        self.reset()


    def first_ob(self):
        return self.states[self.t]

    def step(self,w1,w2):
        if self.FLAG:
            not_terminal = 1
            price = self.price_history[self.t]
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            std = self.states[self.t - 1][0].std(axis=0, ddof=0)
            w2_std = (w2[0]* std).sum()

            #adding risk
            gamma=0.00
            risk=gamma*w2_std

            r = (np.dot(w2, price)[0] - mu)[0]


            reward = np.log(r + eps)

            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == len(self.states) - 1:
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward, 'continue': not_terminal, 'next state': self.states[self.t],
                    'weight vector': w2, 'price': price,'risk':risk}
            return info
        else:
            info = {'reward': 0, 'continue': 1, 'next state': self.states[self.L + 1],
                    'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),
                    'price': self.price_history[self.L + 1],'risk':0}

            self.FLAG=True
            return info

    def reset(self):
        self.t=self.L+1
        self.FLAG = False




        