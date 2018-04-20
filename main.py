import EnvOps
import AnalysisOps
from AnalysisOps import phi
import LogOps
import model
import time
import os,sys
import numpy as np
import tensorflow as tf
from selenium import webdriver

global count
count = 0


def epsodic(env):
    q_model = model.FeedModel()
    q_model.restore_model('model/RL.ckpt')
    sample_path = LogOps.LogOps('memory/eps.csv')
    if os.path.exists('memory/eps.csv') is False:
        sample_path.create()
    global count
    for _ in range(1,1000):
        A = [1]
        while bool(A):
            count = count + 1
            s_old = phi(env.state())
            a = eps_greedy(s = s_old, model= q_model,eps= 0.01)
            env.action(a)
            r = env.reward()
            time.sleep(0.1)
            s_new = phi(env.state())
            sample_path.write(s_old, a, r, s_new)
            A = AnalysisOps.A(s_new)
            if count % 100 == 0:
                s,a,r,s_ = sample_path.reconstruct(1024)
                q_model.DQN_train(s, a, r, s_)
                q_model.save_model('model/RL.ckpt')
        env.retry()
    q_model.sess.close()




def eps_greedy(s,model,eps):
    A = AnalysisOps.A(s)
    if np.random.rand()<eps:
        seed = np.random.randint(12)
        a = A[seed % len(A)]
    else:
        a, _ = model.argmax_a_q(s)
    return a





if __name__ == "__main__":
    browser = webdriver.Chrome('D:\chromedriver_win32\chromedriver.exe')
    env = EnvOps.env(browser)
    env.activate()
    epsodic(env)

