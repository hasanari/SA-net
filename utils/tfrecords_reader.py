import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)

    image.set_shape([224*224*3])
    segmentation.set_shape([224*224*1])

    image = tf.reshape(image,[224,224,3])
    segmentation = tf.reshape(segmentation,[224,224])

    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask

def read_and_decode_non_image(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.float32)  
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)
    #sess = tf.InteractiveSession()
    image.set_shape([224*224*7])
    segmentation.set_shape([224*224*1])

    image = tf.reshape(image,[224,224,7])
    segmentation = tf.reshape(segmentation,[224,224])

    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask




def read_and_decode_non_image_filename(filename_queue):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
            'filename_raw': tf.FixedLenFeature([], tf.string),
            'i_width': tf.FixedLenFeature([], tf.string),
            'i_height': tf.FixedLenFeature([], tf.string)
        }

    )
    # must be read back as uint8 here
    
    print( features['i_height'] )
    i_height = 600
    i_width = 800
    filename = features['filename_raw'] 
    image = tf.decode_raw(features['image_raw'], tf.float32)  
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)

    image.set_shape([i_height*i_width*7])
    segmentation.set_shape([i_height*i_width*1])

    image = tf.reshape(image,[i_height,i_width,7])
    segmentation = tf.reshape(segmentation,[i_height,i_width])

    rgb = tf.cast(image, tf.float32)

    mask = tf.cast(segmentation, tf.float32)
    mask = tf.cast(mask, tf.int64)
    
    return rgb, mask, filename

    
def input_pipeline(filenames, batch_size, num_epochs, image_only=True, return_filename=False):
    
    _filenames = filenames.split(",")
    filename_queue = tf.train.string_input_producer(
        _filenames, num_epochs=num_epochs,shuffle=True)

    
    if(return_filename):

        image, label, filename = read_and_decode_non_image_filename(filename_queue)
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        images_batch, labels_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        
        return images_batch, labels_batch, filename_batch

    else:
        if(image_only):
            image, label = read_and_decode(filename_queue)
        else:    
            image, label = read_and_decode_non_image(filename_queue)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            enqueue_many=False, shapes=None,
            allow_smaller_final_batch=True,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        


        return images_batch, labels_batch