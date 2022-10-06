import os,cv2
import tensorflow as tf
tf.reset_default_graph()   
import numpy as np
import easydict,csv
from glob import glob
import tensorflow.contrib.slim as slim
from tensorflow.keras import layers

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

args = easydict.EasyDict({
    'indir_dataset': "./data",
    'indir_image':"./data/test_image/*.png",
    'indir_checkpoint': './result/ckpt/latest_model.ckpt',
    'T': 32,
    'H': 256,
    'W': 256,
    'C': 3,
    'oudir_pred': "./result/pred"
})

class_names_list, label_values = get_label_info(os.path.join(args.indir_dataset, "class_dict.csv"))
num_classes = len(label_values)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
net_input = tf.placeholder(tf.float32,shape=[None, None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None, None,None,None,num_classes]) 
network = build(net_input=net_input,num_classes=num_classes)
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
saver.restore(sess, args.indir_checkpoint)
if not os.path.isdir(args.oudir_pred):
    os.makedirs(args.oudir_pred)

## Testing
ls = glob(args.indir_image)
N = len(ls)
epoch = N//args.T   
for i in range(epoch):
  input_image = np.zeros((args.T, args.H, args.W, args.C), dtype=np.float32) 
  for j in range(args.T):
      loaded_image = cv2.imread(ls[j+i*args.T])
      loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)  
      input_image[j, :, :, :] = loaded_image
  input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0   
  output_image = sess.run(network,feed_dict={net_input:input_image})  
  output_image = np.array(output_image[0,:,:,:,:])   
  output_image = np.argmax(output_image, axis = -1)
  for j in range(args.T):
      out_vis_image = colour_code_segmentation(output_image[j,:,:], label_values)  
      out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
      name = args.oudir_pred +'/'+ str(j+i*args.T) + '.png'
      cv2.imwrite(name, out_vis_image)
      
if epoch*args.T<N:
    remain = N-epoch*args.T
    input_image = np.zeros((args.T, args.H, args.W, args.C), dtype=np.float32) 
    for j in range(args.T):
      loaded_image = cv2.imread(ls[N-args.T+j])
      loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)  
      input_image[j, :, :, :] = loaded_image
    input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0  
    output_image = sess.run(network,feed_dict={net_input:input_image})  
    output_image = np.array(output_image[0,:,:,:,:])   
    output_image = np.argmax(output_image, axis = -1)
    for j in range(remain):
        out_vis_image = colour_code_segmentation(output_image[args.T-remain+j,:,:], label_values)  
        out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
        name = args.oudir_pred +'/'+ str(j+epoch*args.T) + '.png'
        cv2.imwrite(name, out_vis_image)  