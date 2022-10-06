from __future__ import print_function
import os,time,cv2
import tensorflow as tf
tf.reset_default_graph()    
import tensorflow.contrib.slim as slim
from tensorflow.keras import layers
import numpy as np
import argparse,csv
import random
from operator import add

def get_label_info(csv_path):
    filename, file_extension = os.path.splitext(csv_path)
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

def load_image(name_list, ind, T):
    re=[]
    for i in range(T):
        img = cv2.cvtColor(cv2.imread(name_list[ind+i],-1), cv2.COLOR_BGR2RGB)
        re.append(img)
    return re

def one_hot_it(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x

def build(net_input, num_classes):  
    net = slim.conv3d(net_input, 48, [2, 2, 2], activation_fn=None)          
    net = tf.nn.relu(net)
    net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], padding='SAME')  
    net = slim.conv3d(net, 96, [2, 2, 2], activation_fn=None)                
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)    
    enc1=net
    net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], padding='SAME')  
    net = slim.conv3d(net, 192, [2, 2, 2], activation_fn=None)               
    net = tf.nn.relu(net)
    enc2=net
    net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], padding='SAME')  
    net = slim.conv3d(net, 384, [2, 2, 2], activation_fn=None)               
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    enc3=net
    net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], padding='SAME')   
    net = slim.conv3d(net, 384, [2, 2, 2], activation_fn=None)                
    net = tf.nn.relu(net)   
    enc4=net    
    net = slim.max_pool3d(net, [2, 2, 2], stride=[2, 2, 2], padding='SAME')   
    net = slim.conv3d(net, 384, [2, 2, 2], activation_fn=None)                
    net = tf.nn.relu(net)

    net = layers.UpSampling3D(size=(2,2,2))(net)
    net = tf.concat([net, enc4], axis=-1)                              
    net = slim.conv3d(net, 192, [2, 2, 2], activation_fn=None)         
    net = tf.nn.relu(net)
    net = layers.UpSampling3D(size=(2,2,2))(net)
    net = tf.concat([net, enc3], axis=-1)                              
    net = slim.conv3d(net, 192, [2, 2, 2], activation_fn=None)         
    net = tf.nn.relu(net)
    net = layers.UpSampling3D(size=(2,2,2))(net)                                        
    net = tf.concat([net, enc2], axis=-1)                              
    net = slim.conv3d(net, 192, [2, 2, 2], activation_fn=None)       
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)
    net = layers.UpSampling3D(size=(2,2,2))(net)                                       
    net = tf.concat([net, enc1], axis=-1)                              
    net = slim.conv3d(net, 96, [2, 2, 2], activation_fn=None)        
    net = tf.nn.relu(net)
    net = layers.UpSampling3D(size=(2,2,2))(net)                                        
    net = slim.conv3d(net, 48, [2, 2, 2], activation_fn=None)          
    net = slim.batch_norm(net, fused=True)
    net = tf.nn.relu(net)   
    net = slim.conv3d(net, num_classes, [1, 1, 1], activation_fn=None, scope='logits')   
    return net

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--epoch_start_i', type=int, default=0)
parser.add_argument('--checkpoint_step', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_val_images', type=int, default=5)
parser.add_argument('--validation_step', type=int, default=1)
parser.add_argument('--indir_dataset', type=str, default="./data")
parser.add_argument('--oudir_validation', type=str, default='./result/validation')
parser.add_argument('--oudir_checkpoint', type=str, default='./result/ckpt')
parser.add_argument('--T', type=int, default=32)
parser.add_argument('--H', type=int, default=256)
parser.add_argument('--W', type=int, default=256)
parser.add_argument('--C', type=int, default=3)
args = parser.parse_args(args=[])
T = args.T
H = args.H
W = args.W
C = args.C

class_names_list, label_values = get_label_info(os.path.join(args.indir_dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name
num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,None,3])  
net_output = tf.placeholder(tf.float32,shape=[None,None,None,None,num_classes])
network = build(net_input=net_input, num_classes=num_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

input_names,output_names = prepare_data(dataset_dir=args.indir_dataset)
N = len(input_names)
all_id_list = np.arange(0, N)

# Training 
for epoch in range(args.epoch_start_i, args.num_epochs):
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    val_id_list = random.sample(range(0, N-T), args.num_val_images)
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
     
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)
    flag=0
    for k in range(len(val_input_names)):
        if val_indices[k]>N-1-T:
            val_indices[k]=N-1-T - flag
            flag += 1
 
    val_out_vis_dir = args.oudir_validation + '/' + 'val_out_vis_image_' + str(epoch)
    if not os.path.isdir(val_out_vis_dir):
        os.makedirs(val_out_vis_dir)
    
    num_iters = int(np.floor(N / args.batch_size))
    for i in range(num_iters-T):
        input_image_batch = []
        output_image_batch = []
                
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = all_id_list[index]
            
            input_images = load_image(input_names,id, T)   
            output_images = load_image(output_names, id, T)

            with tf.device('/cpu:0'):
                input_image = np.zeros((T, H, W, C), dtype=np.float32)   
                output_image = np.zeros((T, H, W, num_classes), dtype=np.float32)  
                for k in range(T):
                    input_image[k,:,:,:] = np.float32(input_images[k]) / 255.0   
                    output_image[k,:,:,:] = np.float32(one_hot_it(label=output_images[k], label_values=label_values))   
                input_image_batch.append(np.expand_dims(input_image, axis=0))  
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        
    model_checkpoint_name = args.oudir_checkpoint + '/latest_model'+ '.ckpt'
    saver.save(sess,model_checkpoint_name)
    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        saver.save(sess,"%s/%04d/model.ckpt"%(args.oudir_checkpoint,epoch))
    
    # Validation
    if epoch % args.validation_step == 0:
        for ind in val_indices:
            input_images = load_image(input_names,ind,T) 
            input_image = np.zeros((T, H, W, C), dtype=np.float32)
            for k in range(len(input_images)):
                input_image[k,:,:,:] = np.float32(input_images[k]) / 255.0
            input_image = np.expand_dims(input_image, axis=0)  
            output_image = sess.run(network,feed_dict={net_input:input_image})  
            output_image = np.array(output_image[0,:,:,:])  
            output_image = np.argmax(output_image, axis = -1)
            file_name = os.path.basename(input_names[ind])
            for k in range(T):
                output_vis=colour_code_segmentation(output_image[k,:,:], label_values) 
                cv2.imwrite(val_out_vis_dir + '/val' + str(k) + '_' + file_name, cv2.cvtColor(np.uint8(output_vis), cv2.COLOR_RGB2BGR))
        
