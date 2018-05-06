import tensorflow as tf
import numpy as np
import tflearn
import pandas as pd
import LogOps
import AnalysisOps
import operator





# Learning Rate Parameters
INIT_LEARNING_RATE = 1e-4
LR_DECAY_PER_100K = 0.98
# pylint: disable=too-many-instance-attributes,too-few-public-methods
class FeedModel(object):
  """Class to construct and collect all relevant tensors of the model."""
  def __init__(self):
      self.state_batch_placeholder = tf.placeholder(
          tf.float32, shape=(None, 4, 4, 1))
      self.targets_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
      self.actions_placeholder = tf.placeholder(tf.float32, shape=(None, 4))

      self.pred, self.loss, self.optimizer = inference_graph(self.state_batch_placeholder,
                                                             self.targets_placeholder,
                                                             self.actions_placeholder)
      self.sess = tf.Session()
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
      self.tic = 0

  def restore_model(self,model_path):
      self.saver.restore(self.sess, model_path)
  def save_model(self,model_path):
      self.saver.save(self.sess, model_path)
  def q_value(self,state):
      Table = AnalysisOps.Table()

      state = state.reshape(4, 4, 1)
      A = AnalysisOps.A(state)

      q = self.sess.run(self.pred, feed_dict={self.state_batch_placeholder : [state for _ in A],
                             self.actions_placeholder : np.array([Table.encode(a) for a in A])})
      return {a:q[i][0] for i,a in enumerate(A)}

  def argmax_a_q(self,state):
      table = AnalysisOps.Table()
      q_table = self.q_value(state)

      action = max(q_table.items(), key=operator.itemgetter(1))[0]
      action_encoded = table.encode(action)

      return action, action_encoded

  def DQN_train(self, s_old, a, r, s_new):
      mini_batch = s_old.shape[0]
      y = r
      for i in range(0, mini_batch):
          if not AnalysisOps.A(s_new[i]):
              pass
          else:
              correct = max(self.q_value(s_new[i]).values())
              y[i] = y[i] + correct
      L = 0
      for _ in range(0,5):
          self.sess.run(self.optimizer, feed_dict={self.state_batch_placeholder : s_old.tolist(),
                                           self.actions_placeholder : a.tolist(),
                                           self.targets_placeholder : y.tolist()})
          loss = self.sess.run(self.loss, feed_dict={self.state_batch_placeholder : s_old.tolist(),
                                           self.actions_placeholder : a.tolist(),
                                           self.targets_placeholder : y.tolist()})
          if np.abs(L - loss)<10e-3:
              break
          else:
              L = loss
          print('Loss = %f'%loss)


def inference_graph(state_batch_placeholder,target_placeholder,actions_placeholder):
    conv_a = tflearn.conv_2d(incoming = state_batch_placeholder, nb_filter = 1024,
                             filter_size=(2, 1), strides = 1,padding='valid', activation='relu')
    conv_b = tflearn.conv_2d(incoming = state_batch_placeholder, nb_filter = 1024,
                             filter_size=(1, 2), strides = 1,padding='valid', activation='relu')

    conv_aa = tflearn.conv_2d(incoming = conv_a, nb_filter = 4096,
                              filter_size=(2, 1), strides = 1, padding='valid', activation='relu')
    conv_ab = tflearn.conv_2d(incoming = conv_a, nb_filter = 4096,
                              filter_size=(1, 2), strides = 1, padding='valid', activation='relu')
    conv_ba = tflearn.conv_2d(incoming=conv_b, nb_filter=4096,
                              filter_size=(2, 1), strides = 1, padding='valid', activation='relu')
    conv_bb = tflearn.conv_2d(incoming=conv_b, nb_filter=4096,
                              filter_size=(1, 2), strides = 1, padding='valid', activation='relu')
    action_fc = tflearn.fully_connected(incoming = actions_placeholder, n_units = 1024,
                                        activation = 'elu')
    concat_layer = tflearn.merge([tflearn.flatten(layer) for layer in [conv_aa, conv_ab, conv_ba, conv_bb,
                                                                       conv_a, conv_b, action_fc]],
                                 mode = 'concat')
    normaliize = tflearn.batch_normalization(incoming = concat_layer)

    pred = tflearn.fully_connected(incoming = normaliize, n_units = 1, activation= 'relu')

    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = target_placeholder,
                                                       predictions = pred))

    optimizer = tf.train.AdamOptimizer(learning_rate = 10e-5).minimize(loss = loss)
    return pred, loss, optimizer


