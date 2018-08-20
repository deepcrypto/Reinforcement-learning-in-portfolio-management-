# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import json
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import math
from decimal import Decimal
import matplotlib.pyplot as plt
from agents.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

eps=10e-8
epochs=0
M=0

class StockTrader():
    def __init__(self):
        self.reset()

    def reset(self):
        self.wealth = 10e3
        self.total_reward = 0
        self.ep_ave_max_q = 0
        self.loss = 0
        self.actor_loss=0

        self.wealth_history = []
        self.r_history = []
        self.w_history = []
        self.p_history = []

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(M))

    def update_summary(self,loss,r,q_value,actor_loss,w,p):
        self.loss += loss
        self.actor_loss+=actor_loss
        self.total_reward+=r
        self.ep_ave_max_q += q_value
        self.r_history.append(r)
        self.wealth = self.wealth * math.exp(r)
        self.wealth_history.append(self.wealth)
        self.w_history.extend([','.join([str(Decimal(str(w0)).quantize(Decimal('0.00'))) for w0 in w.tolist()[0]])])
        self.p_history.extend([','.join([str(Decimal(str(p0)).quantize(Decimal('0.000'))) for p0 in p.tolist()])])

    def write(self,epoch):
        wealth_history = pd.Series(self.wealth_history)
        r_history = pd.Series(self.r_history)
        w_history = pd.Series(self.w_history)
        p_history = pd.Series(self.p_history)
        history = pd.concat([wealth_history, r_history, w_history, p_history], axis=1)
        history.to_csv('result' + str(epoch) + '-' + str(math.exp(np.sum(self.r_history)) * 100) + '.csv')

    def print_result(self,epoch,agent):
        self.total_reward=math.exp(self.total_reward) * 100
        print('*-----Episode: {:d}, Reward:{:.6f}%,  ep_ave_max_q:{:.2f}, actor_loss:{:2f}-----*'.format(epoch, self.total_reward,self.ep_ave_max_q,self.actor_loss))
        agent.write_summary(self.loss, self.total_reward,self.ep_ave_max_q,self.actor_loss, epoch)
        agent.save_model(epoch)

    def plot_result(self):
        pd.Series(self.wealth_history).plot()
        plt.show()

    def action_processor(self,a,ratio):
        a = np.clip(a + self.noise() * ratio, 0, 1)
        a = a / (a.sum() + eps)
        return a

def parse_info(info):
    return info['reward'],info['continue'],info[ 'next state'],info['weight vector'],info ['price'],info['risk']


def traversal(stocktrader,agent,env,epoch,noise_flag,framework,method,trainable):
    info = env.step(None,None)
    r,contin,s,w1,p,risk=parse_info(info)
    contin=1
    while contin:
        w2 = agent.predict(s)
        if noise_flag=='True':
            w2=stocktrader.action_processor(w2,(epochs-epoch)/epochs)
        env_info = env.step(w1, w2)
        r, contin, s_next, w1, p,risk = parse_info(env_info)

        agent.save_transition(s, w2, r-risk, contin, s_next, w1)
        loss, q_value,actor_loss=0,0,0

        if framework=='DDPG':
            if trainable=="True":
                agent_info= agent.train(method,epoch)
                loss, q_value=agent_info["critic_loss"],agent_info["q_value"]
                if method=='model_based':
                    actor_loss=agent_info["actor_loss"]

        elif framework=='PPO':
            if not contin and trainable=="True":
                agent_info = agent.train(method, epoch)
                loss, q_value = agent_info["critic_loss"], agent_info["q_value"]
                if method=='model_based':
                    actor_loss=agent_info["actor_loss"]

        stocktrader.update_summary(loss,r,q_value,actor_loss,w2,p)
        s = s_next





def parse_config(config,mode):
    codes = config["session"]["codes"]
    start_date = config["session"]["start_date"]
    end_date = config["session"]["end_date"]
    features = config["session"]["features"]
    agent_config = config["session"]["agents"]
    market = config["session"]["market_types"]
    noise_flag, record_flag, plot_flag=config["session"]["noise_flag"],config["session"]["record_flag"],config["session"]["plot_flag"]
    predictor, framework, window_length = agent_config
    reload_flag, trainable=config["session"]['reload_flag'],config["session"]['trainable']
    method=config["session"]['method']

    global epochs
    epochs = int(config["session"]["epochs"])

    if mode=='test':
        record_flag='True'
        noise_flag='False'
        plot_flag='True'
        reload_flag='True'
        trainable='False'
        method='model_free'

    print("*--------------------Training Status-------------------*")
    print('Codes:',codes)
    print("Date from",start_date,' to ',end_date)
    print('Features:',features)
    print("Agent:Noise(",noise_flag,')---Recoed(',noise_flag,')---Plot(',plot_flag,')')
    print("Market Type:",market)
    print("Predictor:",predictor,"  Framework:", framework,"  Window_length:",window_length)
    print("Epochs:",epochs)
    print("Trainable:",trainable)
    print("Reloaded Model:",reload_flag)
    print("Method",method)
    print("Noise_flag",noise_flag)
    print("Record_flag",record_flag)
    print("Plot_flag",plot_flag)


    return codes,start_date,end_date,features,agent_config,market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method

def session(config,mode):
    from data.environment import Environment
    codes, start_date, end_date, features, agent_config, market,predictor, framework, window_length,noise_flag, record_flag, plot_flag,reload_flag,trainable,method=parse_config(config,mode)
    env = Environment(start_date, end_date, codes, features, int(window_length),market)


    global M
    M=len(codes)+1

    if framework == 'DDPG':
        print("*-----------------Loading DDPG Agent---------------------*")
        from agents.ddpg import DDPG
        agent = DDPG(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)

    elif framework == 'PPO':
        print("*-----------------Loading PPO Agent---------------------*")
        from agents.ppo import PPO
        agent = PPO(predictor, len(codes) + 1, int(window_length), len(features), '-'.join(agent_config), reload_flag,trainable)

    stocktrader=StockTrader()

    if mode=='train':

        print("Training with {:d}".format(epochs))
        for epoch in range(epochs):
            print("Now we are at epoch", epoch)
            traversal(stocktrader,agent,env,epoch,noise_flag,framework,method,trainable)

            if record_flag=='True':
                stocktrader.write(epoch)

            if plot_flag=='True':
                stocktrader.plot_result()

            stocktrader.print_result(epoch,agent)
            stocktrader.reset()

    elif mode=='test':
        traversal(stocktrader, agent, env, 1, noise_flag,framework,method,trainable)
        stocktrader.write(1)
        stocktrader.plot_result()
        stocktrader.print_result(1, agent)

def build_parser():
    parser = ArgumentParser(description='Provide arguments for training different DDPG or PPO models in Portfolio Management')
    parser.add_argument("--mode",dest="mode",help="download(China), train, test",metavar="MODE", default="train",required=True)
    parser.add_argument("--model",dest="model",help="DDPG,PPO",metavar="MODEL", default="DDPG",required=False)
    return parser


def main():
    parser = build_parser()
    args=vars(parser.parse_args())
    print(args)
    with open('config.json') as f:
        config=json.load(f)
        if args['mode']=='download':
            from data.download_data import DataDownloader
            data_downloader=DataDownloader(config)
            data_downloader.save_data()
        else:
            session(config,args['mode'])

if __name__=="__main__":
    main()