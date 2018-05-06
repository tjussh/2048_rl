import model as md
import os
a = md.FeedModel()
os.mkdir('model')
os.mkdir('memory')
a.restore_model('model/RL.ckpt')
a.sess.run(a.init)
a.save_model('model/RL.ckpt')