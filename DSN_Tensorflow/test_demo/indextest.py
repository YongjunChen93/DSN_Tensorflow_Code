import tensorflow as tf 
import numpy as np
sess = tf.Session()
#Coordinate=tf.constant([[0,1,1,0],[1,2,1,3],[0,3,1,2]])
Coordinate=tf.multiply(tf.ones([1,1600],tf.int32),1)
Temp = tf.unstack(Coordinate)
B = tf.unstack(tf.reshape(Coordinate,[-1,len(Temp)]))
Batch_index = []
for i in range(len(Temp)):
	Batch_index.append(i)
	print(i)
	
Batch_index = tf.transpose(tf.expand_dims(tf.stack(Batch_index),0))
Batch_index = tf.cast(Batch_index,tf.int32)
Batch_index = tf.unstack(Batch_index)
Be_insert = tf.unstack(Batch_index)

for Batch_size in range(len(B)-1):
	for index in range(len(Be_insert), 0, -1):
		Batch_index.insert(index*(Batch_size+1), Be_insert[index-1])	
Batch_index = tf.stack(Batch_index)
Coordinate = tf.reshape(Coordinate,[-1,1])

index = tf.concat([Batch_index,Coordinate],1)
index = tf.cast(index,tf.int64)
U=tf.random_uniform([1,40,40,1],minval=0,maxval=130)
Batch = tf.shape(U)[0]
Batch=tf.cast(Batch,tf.int64)
C = tf.shape(U)[3]
C=tf.cast(C,tf.int64)
height = tf.shape(U)[1]
H=tf.cast(height,tf.int64)
width = tf.shape(U)[2]
W=tf.cast(width,tf.int64)
U=tf.reshape(U,[-1])
U_org=tf.multiply(tf.ones([1,40,40,1]),10)
#Value from U
Sparse_values=tf.SparseTensor(indices=index, values=U, dense_shape=[Batch,H*W*C])
value_from_U=tf.sparse_tensor_to_dense(sp_input=Sparse_values,default_value=-10,validate_indices=False)
value_from_U=tf.cast(value_from_U,'float32')
thred=tf.subtract(tf.ones_like(value_from_U,'float32'),8)
#Select_Or_Not_bool
S_O_R_bool=tf.Tensor.__ge__(value_from_U,thred)
S_O_R_value=tf.cast(S_O_R_bool,tf.float32)
S_O_R_value=tf.subtract(tf.ones_like(S_O_R_value),S_O_R_value)
#Value from U_org
B_org = tf.shape(U_org)[0]
U_org=tf.reshape(U_org,[B_org,-1])
U_org = tf.cast(U_org,'float32')
value_from_U_org = tf.multiply(S_O_R_value,U_org)
#Use to offset the thred value
equal_to_thred=tf.multiply(S_O_R_value,10)
#Ouput value
output = tf.add(tf.add(value_from_U,value_from_U_org),equal_to_thred)
print(sess.run(tf.shape(output)))
print(sess.run(output))

