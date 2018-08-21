# Reinforcement learning in portfolio management

## Introduction

Motivated by "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" by [Jiang et. al. 2017](https://arxiv.org/abs/1706.10059) [1]. In this project:
+ Implement two state-of-art continous deep reinforcement learning algorithms, Deep Deterministic Policy Gradient (DDPG) and Proximal Policy Optimization(PPO) in portfolio management. 
+ Experiments on different settings, such as changing learning rates, optimizers, neutral network structures, China/America Stock data, initializers, noise, features to figure out their influence on the RL agents' performance(cumulative return).
## Using the environment

The environment provides supports for easily testing different reinforcement learning in portfolio management.
+ main.py -  the entrance to run our training and testing framework
+ ./saved_network- contains different saved models after training, with DDPG and PPO sub folders
+ ./summary- contains summaries, also with DDPG and PPO sub folder
+ ./agent- contains ddpg.py, ppo.py and ornstein_uhlenbeck.py(the noise we add to agent's actions during training)
+ ./data- contains America.csv for USA stock data, China.csv for China stock data. download_data.py can download China stock data by Tushare. environment.py generates states data for trading agents.
+ config.json- the configuration file for training or testing settings:
```
{
	"data":{
		"start_date":"2015-01-01",
		"end_date":"2018-01-01",
		"market_types":["stock"],
		"ktype":"D"
	},
	"session":{
		"start_date":"2015-01-05",
		"end_date":"2017-01-01",
		"market_types":"America",
	    "codes":["AAPL","ADBE","BABA","SNE","V"],
		"features":["close"],
		"agents":["CNN","DDPG","3"],
		"epochs":"10000",
		"noise_flag":"False",
		"record_flag":"False",
		"plot_flag":"False",
		"reload_flag":"False",
		"trainable":"True",
		"method":"model_free"
	}
}
```

Download stock data in shenzhen and shanghai stock market in the given period in Day(D) frequency. Options: hours, minutes
```
python main.py --mode=download_data
```
Training/Testing
+ noise_flag=True: actions produced by RL agents are distorted by adding UO noise.
+ record_flag=True: trading details would be stored as a csv file named by the epoch and cumulative return each epoch.
+ plot_flag=True: the trend of wealth would be plot each epoch.
+ reload_flag=True: tensorflow would search latest saved model in ./saved_network and reload.
+ trainable=True: parameters would be updated during each epoch.
+ method=model_based: the first epochs our agents would try to imitate a greedy strategy to quickly improve its performance. Then it would leave it and continue to self-improve by model-free reinforcement learning.

## Result

![result](/result.jpg)

## Contribution

### Contributors

* ***Zhipeng Liang***
* ***Kangkang Jiang***
* ***Hao Chen***
* ***Junhao Zhu***
* ***Yanran Li***
### Institutions

+ ***AI&FintechLab of Likelihood Technology***
+ ***Sun Yat-sen University***

## Acknowledegment

We would like to say thanks to ***Mingwen Liu*** from ***ShingingMidas Private Fund***, ***Zheng Xie*** and ***Xingyu Fu*** from ***Sun Yat-sen University*** for their generous guidance throughout the project.

## Set up

Python Version

+ ***3.6***

Modules needed

+ ***tensorflow(tensorflow-gpu)***
+ ***numpy*** 
+ ***pandas*** 
+ ***matplotlib***

## Contact

+ liangzhp6@mail2.sysu.edu.cn
+ jiangkk3@mail2.sysu.edu.cn
+ chenhao8@mail2.sysu.edu.cn
+ zhujh25@mail2.sysu.edu.cn
+ liyr8@mail2.sysu.edu.cn