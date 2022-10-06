import os,time,cv2
import numpy as np
from glob import glob
import tensorflow as tf
tf.enable_eager_execution()
import easydict
import csv
from tensorflow.contrib import slim
from tensorflow.keras import layers

def conv(tensor, weights, biases): 
    conv = tf.nn.conv2d(tensor, weights, strides=[1,1,1,1], padding='SAME')
    output = conv + biases
    return output

def relu(tensor):  
    return tf.nn.relu(tensor, name='ReLU')

def pool(tensor, s):
    return slim.pool(tensor, [s, s], stride=[s, s], pooling_type='MAX')

def upsample(input_tensor, s=2):
    output_tensor = layers.UpSampling2D(size=(s,s))(input_tensor)
    return output_tensor
    
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

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

def reverse_one_hot(image):
    return np.argmax(image, axis = -1)

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    return colour_codes[image.astype(int)]

args = easydict.EasyDict({
    'indir_test_files':"./data/test_image/*.png",
    'indir_dataset': "./data",
    'indir_pretrained_weights': "./result/ckpt/latest_model.ckpt",
    'outdir_predition': "./result/pred/",
    'H': 256,
    'W':3840,
    'C':3,
    'stride': 2,
    'T':2160
})
H=args.H
W=args.W
C=args.C
s=args.stride
T=args.T

if not os.path.isdir(args.outdir_predition):
    os.makedirs(args.outdir_predition)
class_names_list, label_values = get_label_info(os.path.join(args.indir_dataset, "class_dict.csv"))
num_classes = len(label_values)
reader = tf.train.NewCheckpointReader(args.indir_pretrained_weights)

w0 = reader.get_tensor('Conv/weights')        
b0 = reader.get_tensor('Conv/biases')       
w1 = reader.get_tensor('Conv_1/weights')      
b1 = reader.get_tensor('Conv_1/biases')       
w2 = reader.get_tensor('Conv_2/weights')      
b2 = reader.get_tensor('Conv_2/biases')       
w3 = reader.get_tensor('Conv_3/weights')      
b3 = reader.get_tensor('Conv_3/biases')       
w4 = reader.get_tensor('Conv_4/weights')      
b4 = reader.get_tensor('Conv_4/biases')       
w5 = reader.get_tensor('Conv_5/weights')      
b5 = reader.get_tensor('Conv_5/biases')       
w6 = reader.get_tensor('Conv_6/weights')      
b6 = reader.get_tensor('Conv_6/biases')       
w7 = reader.get_tensor('Conv_7/weights')      
b7 = reader.get_tensor('Conv_7/biases')       
w8 = reader.get_tensor('Conv_8/weights')      
b8 = reader.get_tensor('Conv_8/biases')       
w9 = reader.get_tensor('Conv_9/weights')      
b9 = reader.get_tensor('Conv_9/biases')       
w10 = reader.get_tensor('Conv_10/weights')    
b10 = reader.get_tensor('Conv_10/biases')     
w11 = reader.get_tensor('Conv_11/weights')    
b11 = reader.get_tensor('Conv_11/biases')     
w12 = reader.get_tensor('Conv_12/weights')    
b12 = reader.get_tensor('Conv_12/biases')     
w13 = reader.get_tensor('Conv_13/weights')    
b13 = reader.get_tensor('Conv_13/biases')     
w14 = reader.get_tensor('Conv_14/weights')   
b14 = reader.get_tensor('Conv_14/biases') 
w15 = reader.get_tensor('Conv_15/weights')   
b15 = reader.get_tensor('Conv_15/biases') 
w16 = reader.get_tensor('logits/weights')   
b16 = reader.get_tensor('logits/biases')    

## testing
re = np.zeros((T, W, C), dtype=np.uint8)
out_vis_image = np.zeros((2, W, C), dtype=np.uint8)
files = glob(args.indir_test_files)
for file in files:    
    tf.reset_default_graph()    
    rp = cv2.imread(file)       
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.float32(rp)/255.0
    for i in range(0, T-H + 1):      
      input = rp[i:i+H, :, :]    
      input = np.expand_dims(input, axis=0)
      skip0 = input                                      
      net = conv(input, w0, b0)                          
      net = relu(net)  
      net = pool(net, s)                 
      skip1 = net                        
      net = conv(net, w1, b1)            
      net = relu(net)  
      net = pool(net, s)                 
      skip2 = net                        
      net = conv(net, w2, b2)            
      net = relu(net)           
      net = pool(net, s)                 
      skip3 = net                        
      net = conv(net, w3, b3)            
      net = relu(net)             
      net = pool(net, s)                  
      skip4 = net                        
      net = conv(net, w4, b4)              
      net = relu(net) 
      net = pool(net, s)                 
      skip5 = net                        
      net = conv(net, w5, b5)                
      net = relu(net) 
      net = pool(net, s)                     
      skip6 = net                        
      net = conv(net, w6, b6)            
      net = relu(net) 
      net = pool(net, s)                 
      skip7 = net                         
      net = conv(net, w7, b7)            
      net = relu(net) 
      net = pool(net, s)   
          
      net = upsample(net, s)                         
      net = tf.concat([net, skip7], axis=-1)         
      net = conv(net, w8, b8)                        
      net = relu(net)  
      net = upsample(net, s)                           
      net = tf.concat([net, skip6], axis=-1)         
      net = conv(net, w9, b9)                        
      net = relu(net)  
      net = upsample(net, s)                           
      net = tf.concat([net, skip5], axis=-1)         
      net = conv(net, w10, b10)                      
      net = relu(net)  
      net = upsample(net, s)                              
      net = tf.concat([net, skip4], axis=-1)         
      net = conv(net, w11, b11)                      
      net = relu(net)  
      net = upsample(net, s)                          
      net = tf.concat([net, skip3], axis=-1)        
      net = conv(net, w12, b12)                      
      net = relu(net)   
      net = upsample(net, s)                          
      net = tf.concat([net, skip2], axis=-1)         
      net = conv(net, w13, b13)                      
      net = relu(net) 
      net = upsample(net, s)                         
      net = tf.concat([net, skip1], axis=-1)         
      net = conv(net, w14, b14)                      
      net = relu(net)  
      net = upsample(net, s)                           
      net = tf.concat([net, skip0], axis=-1)         
      net = conv(net, w15, b15)                      
      net = relu(net) 
      net = conv(net, w16, b16)                      
      net = tf.nn.softmax(net, axis=-1, name=None)                               
      net = net.numpy()                                
      net = net[0, :,:,:]                              
      output_image = np.argmax(net, axis = -1) 
      output_image = cv2.cvtColor(np.uint8(colour_code_segmentation(output_image, label_values)), cv2.COLOR_RGB2BGR)   # [H=2, W=1024, C=3]
      
      if i==T-H:
          re[T-H:T, :, :] = output_image
      else:
          re[i, :, :] = output_image[0, :, :]  
    cv2.imwrite(args.outdir_predition+os.path.basename(file), re)