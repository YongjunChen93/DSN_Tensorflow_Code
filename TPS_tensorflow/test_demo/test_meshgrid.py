import tensorflow as tf
sess = tf.Session()

def meshgrid(height, width,Row_controlP_number,Column_controlP_number):
		with tf.variable_scope('meshgrid'):
			#p:[N x 2] input points
			x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
							tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
			y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
							tf.ones(shape=tf.stack([1, width])))   
			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))
			
			px,py = tf.stack([x_t_flat],axis=2),tf.stack([y_t_flat],axis=2)

			#source control points
			x,y = tf.linspace(-1.,1.,Column_controlP_number),tf.linspace(-1.,1.,Row_controlP_number)
			x,y = tf.meshgrid(x,y)
			xs,ys = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1)))
			cpx,cpy = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2])
			px, cpx = tf.meshgrid(px,cpx);py, cpy = tf.meshgrid(py,cpy)
			
			#Compute distance R
			Rx,Ry = tf.square(tf.subtract(px,cpx)),tf.square(tf.subtract(py,cpy))
			R = tf.add(Rx,Ry)
			
			#cp = tf.transpose(cp)
			R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))

			#Source coordinates
			ones = tf.ones_like(x_t_flat) 
			grid = tf.concat([ones, x_t_flat, y_t_flat,R],0)
			return grid	
Row_controlP_number = 2
Column_controlP_number = 2
T = tf.constant([[[1,2,3,4,5,6,7],[4,5,6,7,8,9,10]]])
print(sess.run(tf.shape(T)))
T = tf.cast(T, 'float32')
grid = meshgrid(28, 28, Row_controlP_number, Column_controlP_number)
grid = tf.expand_dims(grid, 0)
grid = tf.reshape(grid, [-1])
grid = tf.tile(grid, tf.stack([1]))
grid = tf.reshape(grid, tf.stack([1, tf.add(tf.multiply(Row_controlP_number,Column_controlP_number),3), -1]))
T_g = tf.matmul(T, grid)
x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
x_s_flat = tf.reshape(x_s, [-1])
y_s_flat = tf.reshape(y_s, [-1])
print(sess.run(tf.shape(T_g)))
