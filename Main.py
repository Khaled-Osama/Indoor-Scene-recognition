from Data_Utils import *
from model_arch import *

BATCH_SIZE = 32
EPOCHS = 60
LR = 0.0001
tf.reset_default_graph()
train_dataset, valid_dataset = loading_dataset()
#map each element in data set from string(image path) to an imag tensor.
#train_dataset = train_dataset.map(preprocess_image)

#make left right flipping to each image then concatenate with the original dataset.
flipping_dataset = train_dataset
flipping_dataset = flipping_dataset.map(flipping)
train_dataset = train_dataset.concatenate(flipping_dataset)
#resize and normalize each image.
train_dataset = train_dataset.map(resize_image)

# do the same for valid dataset.
valid_dataset = valid_dataset.map(preprocess_image)
valid_dataset = valid_dataset.map(resize_image)
#make iteratot for train and valid datasets , shuffle them and dividing them to batches.
train_dataset = train_dataset.shuffle(640).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_initializable_iterator()

valid_dataset = valid_dataset.repeat().batch(BATCH_SIZE) #631

batch_images, batch_labels = train_iterator.get_next()

valid_iterator = valid_dataset.make_initializable_iterator()
valid_batch_images, valid_batch_labels = valid_iterator.get_next()


batch_images_var, batch_labels_var = batch_images, batch_labels

#calling inception resnet function to calculate predictions for batch training images.
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    predictions, end_points = inception_resnet_v2(inputs=batch_images_var, num_classes=10,
                                                  is_training=True, dropout_keep_prob=0.5)

# predictions = tf.squeeze(predictions, [1, 2])
predictions = tf.cast(predictions, tf.float32)
#calculate the error after applying softmax activation function.
slim.losses.softmax_cross_entropy(predictions, batch_labels_var)
total_loss = slim.losses.get_total_loss()
#Adam optimization
optimizer = tf.train.AdamOptimizer(learning_rate=LR)

# compare between predictions and actual labels which returns 1 if equals and 0 if not equal
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_labels_var, 1))
# calculate training accuracy for a batch.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#the same for validation.
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

#list which have the variables name which we later use them to load from pretrained checkpoint.
variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionResnetV2/Logits',
                                                              'InceptionResnetV2/AuxLogits'])

saver = tf.train.Saver()
restorer = tf.train.Saver(variables_to_restore)

# make one step of training (calculate gradients and update weights) using an assigned optimizer.
train_tensor = slim.learning.create_train_op(total_loss, optimizer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(valid_iterator.initializer)

    #model is exist.
    if os.path.exists('my_model/checkpoint'):
        saver.restore(sess, 'my_model/model.ckpt')
        print(sess.run(valid_accuracy))

    else:
        '''for var in tf.global_variables():
          print(var)'''
        # restore pretrained weights.
        restorer.restore(sess, 'incep_ckpt/inception_resnet_v2_2016_08_30.ckpt')

        #tbc = TensorBoardColab()
        # print(batch_images_var.shape)
        # print(batch_labels_var)
        step = 1
        for i in range(EPOCHS):

            for _ in range(0, 79):  # 147
                _loss, _acc, _val_acc, _val_error, pred = sess.run([train_tensor,
                                                                    accuracy, valid_accuracy,
                                                                    valid_error, valid_pred])

                '''if step % 10 == 0:
                    tbc.save_value("errors", "training_error", step, _loss)
                    tbc.save_value("errors", "validation_error", step, _val_error)
                    tbc.save_value("graph_name", "Training_acc", step, _acc)
                    tbc.save_value("graph_name", "valid_acc", step, _val_acc)'''

                print('step %d' % step)
                print('loss =    ' + str(_loss))
                print('trainin_acc =    ' + str(_acc))
                print('valid acc =    ' + str(_val_acc))
                step += 1
                print(np.argmax(pred, 1))

            saver.save(sess, 'inception_resnet_v2/model.ckpt')
            #tbc.close()