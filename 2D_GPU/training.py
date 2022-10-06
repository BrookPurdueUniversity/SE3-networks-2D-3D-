from __future__ import print_function
import os,cv2
import tensorflow as tf
tf.reset_default_graph()    
import tensorflow.contrib.slim as slim
import numpy as np
import argparse
import random
import csv

def get_label_info(csv_path):
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")
    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values

def prepare_data(dataset_dir):
    input_names=[]
    output_names=[]
    for file in os.listdir(dataset_dir + "/train_image"):
        input_names.append(dataset_dir + "/train_image/" + file)
    for file in os.listdir(dataset_dir + "/train_label"):
        output_names.append(dataset_dir + "/train_label/" + file)
    input_names.sort(),output_names.sort()
    return input_names,output_names

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def one_hot_it(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--checkpoint_step', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='./result/ckpt')
parser.add_argument('--validation_step', type=int, default=1)
parser.add_argument('--dataset', type=str, default="./data")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_val_images', type=int, default=4)
parser.add_argument('--validation_dir', type=str, default='./result/validation')
args = parser.parse_args(args=[])

# reading class information
class_names_list, label_values = get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name
num_classes = len(label_values)

def build(inputs, num_classes):  
    skip0 = inputs                                           
    net = slim.conv2d(inputs, 32, [2,2], padding="SAME")     
    net = tf.nn.relu(net)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip1 = net     
    net = slim.conv2d(net, 48, [2,2], padding="SAME")               
    net = tf.nn.relu(net)  
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip2 = net   
    net = slim.conv2d(net, 96, [2,2], padding="SAME")                
    net = tf.nn.relu(net)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip3 = net    
    net = slim.conv2d(net, 128, [2,2], padding="SAME")               
    net = tf.nn.relu(net)    
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip4 = net    
    net = slim.conv2d(net, 192, [2,2], padding="SAME")               
    net = tf.nn.relu(net)      
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip5 = net    
    net = slim.conv2d(net, 256, [2,2], padding="SAME")               
    net = tf.nn.relu(net) 
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip6 = net    
    net = slim.conv2d(net, 320, [2,2], padding="SAME")               
    net = tf.nn.relu(net) 
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    skip7 = net    
    net = slim.conv2d(net, 384, [2,2], padding="SAME")               
    net = tf.nn.relu(net)
    net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')  
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])                          
    net = tf.concat([net, skip7], axis=-1)               
    net = slim.conv2d(net, 384, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])                            
    net = tf.concat([net, skip6], axis=-1)               
    net = slim.conv2d(net, 320, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])
    net = tf.concat([net, skip5], axis=-1)               
    net = slim.conv2d(net, 256, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])
    net = tf.concat([net, skip4], axis=-1)               
    net = slim.conv2d(net, 192, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])                       
    net = tf.concat([net, skip3], axis=-1)               
    net = slim.conv2d(net, 128, [2,2], padding="SAME")   
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])
    net = tf.concat([net, skip2], axis=-1)               
    net = slim.conv2d(net, 96, [2,2], padding="SAME")    
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])
    net = tf.concat([net, skip1], axis=-1)               
    net = slim.conv2d(net, 48, [2,2], padding="SAME")    
    net = tf.nn.relu(net)
    net = tf.image.resize_bilinear(net, size=[tf.shape(net)[1]*2,  tf.shape(net)[2]*2])
    net = tf.concat([net, skip0], axis=-1)               
    net = slim.conv2d(net, 32, [2,2], padding="SAME")    
    net = tf.nn.relu(net)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')   

    return net

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])   
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])   
network = build(inputs=net_input, num_classes=num_classes)   
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))  
opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])    
saver=tf.train.Saver(max_to_keep=1000)   
sess.run(tf.global_variables_initializer())   
input_names,output_names = prepare_data(dataset_dir=args.dataset)
all_id_list = np.arange(0, len(input_names))  

# Start Training Process  
print('Starting Training !') 
for epoch in range(args.num_epochs):
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    
    val_id_list = random.sample(range(0, len(input_names)), args.num_val_images)
    train_id_list = [id_value for id_value in all_id_list if id_value not in val_id_list]

    for i in range(len(val_id_list)):
      id_random = np.random.randint(0,len(train_id_list))
      if train_id_list[id_random] not in val_id_list:
        tmp = val_id_list[i]
        val_id_list[i] = train_id_list[id_random]
        train_id_list[id_random] = tmp
      else:
        id_random = np.random.randint(0,len(train_id_list))
     
    for val_id in val_id_list:
      val_input_names.append(input_names[val_id])
      val_output_names.append(output_names[val_id])
    val_input_names.sort(), val_output_names.sort()
    for train_id in train_id_list:
      train_input_names.append(input_names[train_id])
      train_output_names.append(output_names[train_id])
    train_input_names.sort(), train_output_names.sort()
     
    # Which validation images do we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)

    # Prepare directory during training  
    result_dir = args.validation_dir
    val_out_vis_dir = result_dir + '/' + 'val_out_vis_image_' + str(epoch)
    if not os.path.isdir(val_out_vis_dir):
        os.makedirs(val_out_vis_dir)
      
    # Start training
    id_list = np.random.permutation(len(train_input_names))
    num_iters = int(np.floor(len(id_list) / args.batch_size))
    for i in range(num_iters):
        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = load_image(train_input_names[id])
            output_image = load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image = np.float32(input_image) / 255.0   
                output_image = np.float32(one_hot_it(label=output_image, label_values=label_values))  

                input_image_batch.append(np.expand_dims(input_image, axis=0))  
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})

    # Model saving     
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir("%s/%04d"%(checkpoint_dir,epoch)):
        os.makedirs("%s/%04d"%(checkpoint_dir,epoch))
    
    model_checkpoint_name = checkpoint_dir + '/latest_model.ckpt'
    saver.save(sess,model_checkpoint_name)
    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(checkpoint_dir,epoch))

    # Validation
    if epoch % args.validation_step == 0:
        for ind in val_indices:
            input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])),axis=0)/255.0      
            output_image = sess.run(network,feed_dict={net_input:input_image})  
            output_image = np.array(output_image[0,:,:,:]) 
            output_image = np.argmax(output_image, axis = -1)
            file_name = os.path.basename(val_input_names[ind])
            output_image = colour_code_segmentation(output_image, label_values)
            cv2.imwrite(val_out_vis_dir + '/' + file_name, cv2.cvtColor(np.uint8(output_image), cv2.COLOR_RGB2BGR))