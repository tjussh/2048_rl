import model as md
a = md.FeedModel()
a.restore_model('model/RL.ckpt')
a.sess.run(a.init)
a.save_model('model/RL.ckpt')