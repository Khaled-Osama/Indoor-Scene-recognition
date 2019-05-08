from Data_Utils import *
from model_arch import *

BATCH_SIZE = 32
EPOCHS = 60
tf.reset_default_graph()
train_dataset, valid_dataset = loading_dataset()

# train_dataset = train_dataset.map(preprocess_image)
flipping_dataset = train_dataset
flipping_dataset = flipping_dataset.map(flipping)
train_dataset = train_dataset.concatenate(flipping_dataset)
train_dataset = train_dataset.map(resize_image)

valid_dataset = valid_dataset.map(preprocess_image)
valid_dataset = valid_dataset.map(resize_image)

train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_initializable_iterator()

valid_dataset = valid_dataset.repeat().batch(BATCH_SIZE)

batch_images, batch_labels = train_iterator.get_next()

valid_iterator = valid_dataset.make_initializable_iterator()
valid_batch_images, valid_batch_labels = valid_iterator.get_next()

train_log_dir = []
root = % pwd
train_log_dir.append(root)
train_log_dir.append('/my_model')
train_log_dir = ''.join(train_log_dir)

log_dir = []
log_dir.append(root)
log_dir.append('/log')
log_dir = ''.join(log_dir)

batch_images_var, batch_labels_var = batch_images, batch_labels

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    predictions, end_points = inception_resnet_v2(inputs=batch_images_var, num_classes=10,
                                                  is_training=True, dropout_keep_prob=0.5)

# predictions = tf.squeeze(predictions, [1, 2])
predictions = tf.cast(predictions, tf.float32)

slim.losses.softmax_cross_entropy(predictions, batch_labels_var)
total_loss = slim.losses.get_total_loss()
optimizer = tf.train.AdamOptimizer(learning_rate=.0001)

# loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=batch_labels)

correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_labels_var, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

valid_image_var, valid_labels_var = valid_batch_images, valid_batch_labels
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    valid_pred, _ = inception_resnet_v2(inputs=valid_image_var,
                                        num_classes=10, is_training=True,
                                        drop_out_training=False,
                                        reuse=True)

# valid_pred = tf.squeeze(valid_pred, [1, 2])
valid_error = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_pred, labels=valid_labels_var)
valid_error = tf.reduce_mean(valid_error)
valid_correct_pred = tf.equal(tf.argmax(valid_pred, 1), tf.argmax(valid_labels_var, 1))
valid_accuracy = tf.reduce_mean(tf.cast(valid_correct_pred, tf.float32))

variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionResnetV2/Logits',
                                                              'InceptionResnetV2/AuxLogits'])

saver = tf.train.Saver()
restorer = tf.train.Saver(variables_to_restore)

train_tensor = slim.learning.create_train_op(total_loss, optimizer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(valid_iterator.initializer)

    if os.path.exists('inception_resnet_v2/checkpoint'):
        saver.restore(sess, 'inception_resnet_v2/model.ckpt')
        print(sess.run(valid_accuracy))

    else:
        '''for var in tf.global_variables():
          print(var)'''
        restorer.restore(sess, 'incep_ckpt/inception_resnet_v2_2016_08_30.ckpt')
        tbc = TensorBoardColab()
        # print(batch_images_var.shape)
        # print(batch_labels_var)
        step = 1
        for i in range(EPOCHS):

            for _ in range(0, 79):  # 147
                _loss, _acc, _val_acc, _val_error, pred = sess.run([train_tensor,
                                                                    accuracy, valid_accuracy,
                                                                    valid_error, valid_pred])

                if step % 10 == 0:
                    tbc.save_value("errors", "training_error", step, _loss)
                    tbc.save_value("errors", "validation_error", step, _val_error)
                    tbc.save_value("graph_name", "Training_acc", step, _acc)
                    tbc.save_value("graph_name", "valid_acc", step, _val_acc)
                print('step %d' % step)
                print('loss =    ' + str(_loss))
                print('trainin_acc =    ' + str(_acc))
                print('valid acc =    ' + str(_val_acc))
                step += 1
                print(np.argmax(pred, 1))

            saver.save(sess, 'inception_resnet_v2/model.ckpt')
            tbc.close()