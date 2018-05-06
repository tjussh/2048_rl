# Reinforcement Learning on 2048

This project is a Deep Reinforcement Learning applied on game 2048.  The development environment is on Ubuntu 16.04 and Python 3.6.  The requirement of packages contain web crawler--selenium, tensorflow, tflearn and pandas.  Before execute, you need to install the selenium chrome driver.  The installation refers to http://selenium-python.readthedocs.io/, then we can start run the script.  Notice if you may need to modify the chrome driver address in the main.py, according to your driver address.
```
git clone https://github.com/yitech/2048_rl.git
cd 2048_rl
python __init__.py #run this only when first execute
python main,py
```


# Introduction

There are some script to support the project, each script can work independently.  Welcome everyone fork the script to address your own reinforcement learning algorithm:

The main idea comes from this project https://github.com/tjwei/2048-NN, and 2048 is exactly a good practice for reinforcement learning. The main algorithm is used Deep Q-Network, use the webdriver to interact with the environment and get feedback. More detail of the Deep Q-Network can refer this [ArXiv](https://arxiv.org/abs/1312.5602)

Claim that the project is NOT perform as well as other 2048 project, the highest block I had got is only 1024, maybe there are still some problems I haven't notice. I will state my work in the following.


# Pipeline

This project contains several parts.  First we need etch a module to interact the environment, one convenient way is programming a control panel with https://gabrielecirulli.github.io/2048/, instead of etch a 2048; and also, we need to crawl the result such like score (as reward) and the gird (as state.  This part is etched in EnvOps.py.

Then dependent on the Deep Q-Network, we need to store each step and call them while training, that's the experiment reply part. I put it on the LogOps.py.  I store in a csv file and call it by pandas.

The model part is https://github.com/tjwei/2048-NN  for reference, but not completely copy.  I modify the input of the net by rewrite the input as a function of state and action to describe the  Q-table.

We need some patches to make the project complete.  For example, we can define the legal movement at each state,  transfer action to one-hot encode after crawl,  how to represent the crawled state, and so on.

Finally, we can combine all above in main.py. The choosen of each action apply epsilon-greedy.  Train the path via experiment reply at every K step. 



# Result

![Here is the Screenshot when playing.](https://github.com/yitech/2048_rl/blob/master/Screenshot.png)
If you have any suggestion or better idea, please contact with me.
Maybe I will revise the after I am more familar with reinforcement learning.

My e-mail: yitech@techie.com

