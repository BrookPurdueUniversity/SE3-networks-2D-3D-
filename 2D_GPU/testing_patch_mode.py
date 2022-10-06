import os,cv2
import tensorflow as tf
import numpy as np
import easydict
tf.reset_default_graph()
from glob import glob
import csv
import tensorflow.contrib.slim as slim
args = easydict.EasyDict({
    'test_image':"./data/test_image/*.png",
    'ckpt_dir': './result/ckpt/latest_model.ckpt',
    'dataset': "./data",
    'predict_dir': "./result/pred/",
    'H': 256,
    'T':2160,
})

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

class_names_list, label_values = get_label_info(os.path.join(args.dataset, "class_dict.csv"))
num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])  
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])  
network = build(inputs=net_input, num_classes=num_classes)  
saver=tf.train.Saver(max_to_keep=1000)  
sess.run(tf.global_variables_initializer())   
saver.restore(sess, args.ckpt_dir)   
predict_dir = args.predict_dir
if not os.path.isdir(predict_dir):
    os.makedirs(predict_dir)

# testing
files = glob(args.test_image)
for file in files:
    rp = cv2.imread(file)
    re = np.zeros_like(rp)
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.expand_dims(np.float32(rp),axis=0)/255.0
    
    for i in range(0, args.T-args.H+1):      
        if i%args.H==0:    
            img = rp[:, i:i+args.H, :, :]
            output_img = sess.run(network,feed_dict={net_input:img})  
            output_img = np.argmax(output_img[0,:,:,:], axis = -1)                   
            output_img = colour_code_segmentation(output_img, label_values)  
            output_img = cv2.cvtColor(np.uint8(output_img), cv2.COLOR_RGB2BGR)  
            re[i:i+args.H, :, :] = output_img
        elif i==args.T-args.H:
            img = rp[:, i:i+args.H, :, :]
            output_img = sess.run(network,feed_dict={net_input:img})               
            output_img = np.argmax(output_img[0,:,:,:], axis = -1)      
            output_img = colour_code_segmentation(output_img, label_values)  
            output_img = cv2.cvtColor(np.uint8(output_img), cv2.COLOR_RGB2BGR)     
            re[args.T-args.H:args.T, :, :] = output_img

    cv2.imwrite(predict_dir + os.path.basename(file), re)        
