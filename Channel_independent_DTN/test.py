# python3.0
'''
@ author Yongjun Chen
'''
# Implement 2D version of Dense Transformer Layer
from Dense_Transformer_Network import *
import numpy as np
import tensorflow as tf
import h5py

def main():
	sess = tf.Session()
	# inputs
	U=tf.linspace(1.0,10.0,3*8*8*3)
	U =tf.reshape(U,[3,8,8,3])
	#print(sess.run(U))
	#network initial
	dtn_input_shape = [3,8,8,3]
	control_points_ratio = 2
	# initial DTN class
	transform = DSN_transformer(dtn_input_shape,control_points_ratio)
	# encoder
	encoder= transform.Encoder(U,U)
	#print("encoder",sess.run(encoder))
	#decoder
	decoder = transform.Decoder(encoder,encoder)
	print(sess.run(decoder))
if __name__ == "__main__":
    main()
