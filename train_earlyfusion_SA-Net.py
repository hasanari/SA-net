from __future__ import division
import tensorflow as tf
import numpy as np
import time

import sys

sys.path.insert(0, './architecture')
import earlyfusion_sa_net

sys.path.insert(0, './utils')
from tfrecords_reader import input_pipeline
from utils import get_data_stats, create_border,  decode_labels, inv_preprocess, prepare_label,  loss_IoU, create_reset_metric



train_record = "./tfrecords/train.tfrecords"
val_record = "./tfrecords/val.tfrecords"


IMG_MEAN = np.array((0,0,0), dtype=np.float32) # Not be used

CHECKPOINT_FN = "./deeplab_resnet.ckpt"
EPOCH_INIT = 0 
TOTAL_TRAINING_DATA = 57036

restore_variables = False # Train from scratch 
IGNORE_LABEL = 0
NUM_CLASSES = num_classes = 8
MOMENTUM = 0.9
UPDATE_OPS_COLLECTION = 'sa_net_update_ops'  # must be grouped with training op

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('key', 'earlyfusion_sa_net_01',
                           "Used to identifiy training setting")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 10, "batch size") 
tf.app.flags.DEFINE_integer('max_epoch', 50, "max epoch")

tf.app.flags.DEFINE_integer('gpuIDX', 99, "GPU ID")



test_batch_size = 10
batch_size = FLAGS.batch_size
_KEY = FLAGS.key
SNAPSHOT_DIR = ".logs/"+FLAGS.key 
EPOCH_MAX = FLAGS.max_epoch * 1000 #nested tfrecords, need manual stop at FLAGS.max_epoch

if __name__ == "__main__":


    # Init .py variable
    NP_IMG_MEAN, NP_IMG_STD, NP_IMG_AMAX, NP_IMG_AMIN = get_data_stats()
    BATCH_SIZE = batch_size 
    IMAGES_PER_EPOCH = TOTAL_TRAINING_DATA // FLAGS.batch_size

    # Init tensorflow variable
    TF_IMG_MEAN = tf.constant(NP_IMG_MEAN, dtype=tf.float32)
    TF_IMG_STD = tf.constant(NP_IMG_STD, dtype=tf.float32)    
    TF_IMG_AMAX = tf.constant(NP_IMG_AMAX, dtype=tf.float32)
    TF_IMG_AMIN = tf.constant(NP_IMG_AMIN, dtype=tf.float32) 
    base_lr = tf.constant(FLAGS.learning_rate)
    mean_loss = tf.placeholder(dtype=tf.float32, shape=())
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    current_learning_rate = tf.placeholder(dtype=tf.float32, shape=())
    is_training = tf.placeholder(tf.bool) 
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    
    #Data loading

    image_training, annotation_training = input_pipeline( train_record ,
                                                    FLAGS.batch_size,
                                                    EPOCH_MAX, False)

    image_validation, annotation_validation , _ = input_pipeline( val_record ,
                                                    test_batch_size,
                                                    EPOCH_MAX, image_only=False, return_filename=True)
    
    
    labels = tf.cond(is_training, lambda:annotation_training, lambda:annotation_validation)
    
    
        
    data_input = tf.cond(is_training,lambda: image_training,lambda: image_validation)
    
    data_input = (data_input -  TF_IMG_MEAN ) /  TF_IMG_STD
    
    _image = data_input[:,:,:,4:7]
    _intensity = data_input[:,:,:,0:1]
    _height = data_input[:,:,:,1:2]
        
    
    x = {'_rgb': _image, '_intensity' : _intensity, '_height' : _height}

    pred, logits = earlyfusion_sa_net.inference(x, is_training = is_training, num_classes = num_classes, num_blocks = [3,4,23,3])

    
    #Calculate lossess
    _logits = tf.nn.softmax(logits) # N, H, W, Class
    _labels = tf.one_hot(labels, depth=num_classes)
    reduced_loss = loss_IoU( _logits,   _labels ,  num_classes)
    

    gt =tf.expand_dims(labels, dim=3)  # tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.not_equal(gt, IGNORE_LABEL), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    

    #The value for each metris is resetted every epoch
    mIoU, update_op, reset_MIoU = create_reset_metric(
                    tf.metrics.mean_iou, 'mean_iou',
                    labels=gt,predictions=pred, num_classes=NUM_CLASSES, weights=weights)
    mean_per_class_accuracy, mean_per_class_accuracy_update_op, reset_MCA = create_reset_metric(
                    tf.metrics.mean_per_class_accuracy, 'mean_per_class_accuracy',
                    labels=gt,predictions=pred, num_classes=NUM_CLASSES, weights=weights)
    accuracy, accuracy_update_op, reset_ACC = create_reset_metric(
                    tf.metrics.accuracy, 'accuracy',
                    labels=gt,predictions=pred,  weights=weights)
    
    
    Images_Summary = tf.py_func(inv_preprocess, [ ( ( _image *TF_IMG_STD[4:7]) + TF_IMG_MEAN[4:7] ) /255 , BATCH_SIZE, IMG_MEAN, [0,1,2]], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [labels, BATCH_SIZE, NUM_CLASSES], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, BATCH_SIZE, NUM_CLASSES], tf.uint8)
    border_black = tf.py_func(create_border, [labels], tf.uint8)

    __loss_value = tf.summary.scalar("loss_value", mean_loss)
    
    with tf.name_scope('training'):
        __img_summary = tf.summary.image('images', 
                                             tf.concat(axis=2, values=[Images_Summary, border_black, labels_summary, border_black, preds_summary]), 
                                             max_outputs=BATCH_SIZE) # Concatenate row-wise. 
        __mIoU = tf.summary.scalar("mIoU", mIoU)
        __mean_per_class_accuracy = tf.summary.scalar("mean_per_class_accuracy", mean_per_class_accuracy)         
        __accuracy = tf.summary.scalar("accuracy", accuracy)

    
    with tf.name_scope('testing'):
        t__img_summary = tf.summary.image('images', 
                                             tf.concat(axis=2, values=[Images_Summary, border_black, labels_summary, border_black, preds_summary]), 
                                             max_outputs=BATCH_SIZE) # Concatenate row-wise. 
        t__mIoU = tf.summary.scalar("mIoU", mIoU)
        t__mean_per_class_accuracy = tf.summary.scalar("mean_per_class_accuracy", mean_per_class_accuracy)         
        t__accuracy = tf.summary.scalar("accuracy", accuracy)
    
    
    training_summary = tf.summary.merge([__mIoU,__mean_per_class_accuracy,__accuracy], name='training_summary')
    testing_summary = tf.summary.merge([t__mIoU,t__mean_per_class_accuracy,t__accuracy,t__img_summary], name='testing_summary')
    

    #This saver is only used for initialization, and cannot be used for additional training. 
    saver = tf.train.Saver([var for var in tf.global_variables() if "scale_fcn" not in var.name])

    #Update using momentum update with CONSTANT initial learning rate
    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(reduced_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #Batchnorm update
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)


    init_global = tf.global_variables_initializer()
    init_locals = tf.local_variables_initializer() # v0.12

    config = tf.ConfigProto( allow_soft_placement = True)
    
    with tf.Session(config=config) as sess:
        
        sess.run([init_global, init_locals])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        # Restore variables
        if restore_variables: 
            
            print("Restoring model..")
            ckpt = tf.train.get_checkpoint_state( SNAPSHOT_DIR )
            # check current saver
            if ckpt and ckpt.model_checkpoint_path:
                
                CHECKPOINT_FN = ckpt.model_checkpoint_path
                EPOCH_INIT = int ( ckpt.model_checkpoint_path.split("-")[-1] ) 
                saver = tf.train.Saver(tf.global_variables())
            
            saver.restore(sess, CHECKPOINT_FN)
            print("Model restored...", CHECKPOINT_FN)

            
        print("EPOCH_INIT", EPOCH_INIT) 
        
        # new saver
        saver = tf.train.Saver(tf.global_variables())
        
        summary_writer = tf.summary.FileWriter(SNAPSHOT_DIR, sess.graph)
        
       
        itr=0
        current_miou = 0
        current_epoch = EPOCH_INIT
        start_time = time.time()         
        means_collect = [] 
        
        try:

            while not coord.should_stop():

                #Training process 
                loss_val, _ = sess.run([reduced_loss,train_op ], feed_dict={step_ph:itr,keep_probability: 0.85, is_training:True})

                assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

                means_collect.append(loss_val)
                
                itr+=1

                if itr % 100 == 0:

                    elapsed=time.time()-start_time

                    print(_KEY+" Epoch: %d, Step: %d, Train_loss:%g; %.3f sec/%d-batch" % (current_epoch, itr, loss_val, elapsed,100*FLAGS.batch_size))
                    start_time = time.time()

                if current_epoch == EPOCH_INIT or itr % IMAGES_PER_EPOCH == 0:

                    itr=0

                    _mean_loss = np.mean(np.asarray(means_collect, dtype=np.float32))
                    
                    means_collect = []
                    

                    #Calculate training accuracy, not really useful for final inference, but helpful for debugging the network
                    _, summary_str = sess.run([train_op, __loss_value], feed_dict={step_ph:itr, keep_probability: 0.85, is_training:True, mean_loss: _mean_loss})
                    
                    if(current_epoch != EPOCH_INIT ):
                        summary_writer.add_summary(summary_str, current_epoch) 
                        summary_writer.flush()
                    
                        sess.run([reset_MIoU, reset_MCA,reset_ACC])

                        for itr_test in range(5000 // FLAGS.batch_size ):                             
                            train_loss, summary_str,mca,_,acc,_ = sess.run([mIoU, update_op,mean_per_class_accuracy, mean_per_class_accuracy_update_op,accuracy, accuracy_update_op ], feed_dict={keep_probability: 0.85, is_training:True})
                            print(_KEY+" Train Step: %d, mIoU:%g, mca:%g, acc:%g" % (itr_test, train_loss, mca,acc))
                            _train_loss = train_loss

                        _,_,_, summary_str = sess.run( [accuracy_update_op, update_op, mean_per_class_accuracy_update_op, training_summary], feed_dict={step_ph:itr,keep_probability: 0.85, is_training:True})
                        summary_writer.add_summary(summary_str, current_epoch)
                        summary_writer.flush()


                    #Calculate validation accuracy
                    sess.run([reset_MIoU, reset_MCA,reset_ACC])
                    for itr_test in range(187 // test_batch_size ):
                        train_loss, summary_str,mca,_,acc,_ = sess.run([mIoU, update_op,mean_per_class_accuracy, mean_per_class_accuracy_update_op,accuracy, accuracy_update_op ], feed_dict={keep_probability: 0.85, is_training:False})
                        print(_KEY+" Test Step: %d, mIoU:%g, mca:%g, acc:%g" % (itr_test, train_loss, mca,acc))
                        _train_loss = train_loss

                    _,_,_, summary_str = sess.run( [accuracy_update_op, update_op, mean_per_class_accuracy_update_op, testing_summary], feed_dict={step_ph:itr,keep_probability: 0.85, is_training:False})
                    summary_writer.add_summary(summary_str, current_epoch)
                    summary_writer.flush()

                    #Saving the best model only
                    if( _train_loss > current_miou ):
                        current_miou = _train_loss
                        saver.save(sess, SNAPSHOT_DIR + "/model.ckpt", current_epoch)
                        
                    current_epoch+=1  
                    

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
                