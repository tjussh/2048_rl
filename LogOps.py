import pandas as pd
import numpy as np
import AnalysisOps


class LogOps:
    column_name = ['t']
    for i in range(0, 16):
        column_name.append('s_old_' + str(i))
    column_name.append('Up')
    column_name.append('Left')
    column_name.append('Down')
    column_name.append('Right')
    column_name.append('reward')
    for i in range(0, 16):
        column_name.append('s_new_' + str(i))

    def __init__(self, SaveAddress):
        self.columns = LogOps.column_name
        self.log = pd.DataFrame(columns = self.columns)
        self.t = 0
        self.SaveAddress = SaveAddress

    def create(self):
        self.log.to_csv(self.SaveAddress,index=None)


    def write(self,s_old, a, r, s_new):
        table = AnalysisOps.Table()

        s_old = np.squeeze(s_old)
        s_old = s_old.flatten()

        action_encoded = table.encode(a)

        s_new = np.squeeze(s_new)
        s_new = s_new.flatten()

        sample = np.hstack((self.t, s_old, action_encoded, r, s_new))
        sample = pd.DataFrame(sample).T
        sample.to_csv(self.SaveAddress, mode='a', index=None, header=None)

        self.t = self.t + 1

    def reconstruct(self,mini_batch):
        X = pd.read_csv(self.SaveAddress)
        size = min([mini_batch,X.shape[0]])
        X = X.sample(n = size)
        s_old = np.array(X.iloc[:, 1:17])
        s_old = s_old.reshape((size, 4, 4, 1))
        action = np.array(X.iloc[:, 17:21])
        action = action.reshape((size, 4))
        reward = np.array(X.iloc[:, 21])
        reward = reward.reshape((size,1))
        s_new = np.array(X.iloc[:, 22:38])
        s_new = s_new.reshape((size, 4, 4, 1))


        return s_old, action, reward, s_new



