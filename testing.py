import tensorflow as tf
from Data_Utils import *
from model_arch import *
import pandas as pd
tf.reset_default_graph()
labels, testing = load_testing_data()
with slim.arg_scope(inception_resnet_v2_arg_scope()):
  preds,_ = inception_resnet_v2(inputs=testing, num_classes=10, is_training=True, drop_out_training=False)
softmax = tf.nn.softmax(preds)
predictions = None
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, 'my_model/model.ckpt')
  predictions = sess.run(softmax)
  print(predictions[0])
  predictions = np.argmax(predictions, 1) + 1
  print(predictions.shape)
  print(labels)
submission = pd.concat([pd.Series(labels, name='id'), pd.Series(predictions, name='label')], axis=1)
submission.to_csv("indoor_scene.csv",index=False)
print(submission)