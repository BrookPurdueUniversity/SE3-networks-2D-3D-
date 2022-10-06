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
    return colour_codes[image.astype(int)]

args = easydict.EasyDict({
    'indir_test_files':"./data/test_image/*.png",
    'indir_dataset': "./data",
    'indir_pretrained_weights': "./result/ckpt/latest_model.ckpt",
    'outdir_predition': "./result/pred/",
    'H': 256,
    'W': 3840,
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

re = np.zeros((T, W, C), dtype=np.uint8)
out_vis_image = np.zeros((2, W, C), dtype=np.uint8)
M0_p = np.zeros([1, 2, W, C], np.float32)                
M0_c = np.zeros([1, 2, W, 32], np.float32)               
M0_c = tf.Variable(M0_c)
M1_p = np.zeros([1, 3, W//2, 32], np.float32)      
M1_p = tf.Variable(M1_p)
M1_c = np.zeros([1, 3, W//2, 48], np.float32)       
M1_c = tf.Variable(M1_c)
M2_p = np.zeros([1, 5, W//4, 48], np.float32)       
M2_p = tf.Variable(M2_p)
M2_c = np.zeros([1, 5, W//4, 96], np.float32)       
M2_c = tf.Variable(M2_c)
M3_p = np.zeros([1, 9, W//8, 96], np.float32)      
M3_p = tf.Variable(M3_p)
M3_c = np.zeros([1, 9, W//8, 128], np.float32)       
M3_c = tf.Variable(M3_c)
M4_p = np.zeros([1, 17, W//16, 128], np.float32)       
M4_p = tf.Variable(M4_p)
M4_c = np.zeros([1, 17, W//16, 192], np.float32)      
M4_c = tf.Variable(M4_c)
M5_p = np.zeros([1, 33, W//32, 192], np.float32)      
M5_p = tf.Variable(M5_p)
M5_c = np.zeros([1, 33, W//32, 256], np.float32)      
M5_c = tf.Variable(M5_c)
M6_p = np.zeros([1, 65, W//64, 256], np.float32)     
M6_p = tf.Variable(M6_p)
M6_c = np.zeros([1, 65, W//64, 320], np.float32)     
M6_c = tf.Variable(M6_c)
M7_p = np.zeros([1, 129, W//128, 320], np.float32)     
M7_p = tf.Variable(M7_p)
M7_c = np.zeros([1, 129, W//128, 384], np.float32)      
M7_c = tf.Variable(M7_c)

## Testing
files = glob(args.indir_test_files)
for file in files: 
    tf.reset_default_graph() 
    rp = cv2.imread(file)              
    rp = cv2.cvtColor(rp, cv2.COLOR_BGR2RGB)
    rp = np.float32(rp)/255.0  
    
    for i in range(0, T):
      M0_p[0][0][:][:] = M0_p[0][1][:][:]
      M0_p[0][-1][:][:] = rp[i][:][:]
      net = tf.nn.relu(conv(M0_p, w0, b0), name='ReLU')                              
      M0_c = tf.concat([tf.slice(M0_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)

      net = slim.pool(M0_c, [s, s], stride=[s, s], pooling_type='MAX')
      M1_p = tf.concat([tf.slice(M1_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net1 = tf.concat([tf.slice(M1_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M1_p, [0,2,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net1, w1, b1), name='ReLU')            
      M1_c = tf.concat([tf.slice(M1_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M1_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M1_c, [0,2,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M2_p = tf.concat([tf.slice(M2_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net2 = tf.concat([tf.slice(M2_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M2_p, [0,4,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net2, w2, b2), name='ReLU')             
      M2_c = tf.concat([tf.slice(M2_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M2_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M2_c, [0,4,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M3_p = tf.concat([tf.slice(M3_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net3 = tf.concat([tf.slice(M3_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M3_p, [0,8,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net3, w3, b3), name='ReLU')            
      M3_c = tf.concat([tf.slice(M3_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M3_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M3_c, [0,8,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M4_p = tf.concat([tf.slice(M4_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net4 = tf.concat([tf.slice(M4_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M4_p, [0,16,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net4, w4, b4), name='ReLU')             
      M4_c = tf.concat([tf.slice(M4_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)      
      net = tf.concat([tf.slice(M4_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M4_c, [0,16,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M5_p = tf.concat([tf.slice(M5_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net5 = tf.concat([tf.slice(M5_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M5_p, [0,32,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net5, w5, b5), name='ReLU')            
      M5_c = tf.concat([tf.slice(M5_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M5_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M5_c, [0,32,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M6_p = tf.concat([tf.slice(M6_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net6 = tf.concat([tf.slice(M6_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M6_p, [0,64,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net6, w6, b6), name='ReLU')            
      M6_c = tf.concat([tf.slice(M6_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M6_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M6_c, [0,64,0,0], [-1,-1,-1,-1])], axis=1)
      
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')
      M7_p = tf.concat([tf.slice(M7_p, [0,1,0,0], [-1,-1,-1,-1]), net], axis=1)
      net7 = tf.concat([tf.slice(M7_p, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M7_p, [0,128,0,0], [-1,-1,-1,-1])], axis=1)     
      net = tf.nn.relu(conv(net7, w7, b7), name='ReLU')           
      M7_c = tf.concat([tf.slice(M7_c, [0,1,0,0], [-1,-1,-1,-1]), tf.slice(net, [0,0,0,0], [-1,1,-1,-1])], axis=1)     
      net = tf.concat([tf.slice(M7_c, [0,0,0,0], [-1,1,-1,-1]), tf.slice(M7_c, [0,128,0,0], [-1,-1,-1,-1])], axis=1)
      net = slim.pool(net, [s, s], stride=[s, s], pooling_type='MAX')

      net = layers.UpSampling2D(size=(s,s))(net)
      net = tf.concat([net, net7], axis=-1)      
      net = tf.nn.relu(conv(net, w8, b8), name='ReLU') 

      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))
      net = tf.concat([net, net6], axis=-1)       
      net = tf.nn.relu(conv(net, w9, b9), name='ReLU')
      
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))  
      net = tf.concat([net, net5], axis=-1)       
      net = tf.nn.relu(conv(net, w10, b10), name='ReLU')
      
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))      
      net = tf.concat([net, net4], axis=-1)      
      net = tf.nn.relu(conv(net, w11, b11), name='ReLU')
      
      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))              
      net = tf.concat([net, net3], axis=-1)     
      net = tf.nn.relu(conv(net, w12, b12), name='ReLU')

      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))      
      net = tf.concat([net, net2], axis=-1)       
      net = tf.nn.relu(conv(net, w13, b13), name='ReLU')

      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))     
      net = tf.concat([net, net1], axis=-1)        
      net = tf.nn.relu(conv(net, w14, b14), name='ReLU')

      net = layers.UpSampling2D(size=(s,s))(tf.slice(net, [0,1,0,0], [-1,-1,-1,-1]))       
      net = tf.concat([net, tf.Variable(M0_p)], axis=-1)       
      net = tf.nn.relu(conv(net, w15, b15), name='ReLU')
      net= tf.nn.softmax(conv(net, w16, b16), axis=-1, name=None)
      net = (tf.squeeze(net, 0)).numpy()                         
      out_image = np.argmax(net, axis = -1)
      out_image = cv2.cvtColor(np.uint8(colour_code_segmentation(out_image, label_values)), cv2.COLOR_RGB2BGR)   
      re[i, :, :] = out_image[1, :, :]  
    cv2.imwrite(args.outdir_predition+os.path.basename(file), re)