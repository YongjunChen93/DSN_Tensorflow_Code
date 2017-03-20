import tensorflow as tf
sess = tf.Session()
def makeT(cp, Row_P_N, Column_P_N):
	cp = tf.reshape(cp,(-1,2,16))
	N_f = 1
	x,y = tf.linspace(-1.,1.,Column_P_N),tf.linspace(-1.,1.,Row_P_N)
	x,y = tf.meshgrid(x,y)
	xs,ys = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1)))
	cp_s = tf.concat([xs,ys],0)
	cp_s_trans = tf.transpose(cp_s)
	##===Compute distance R
	xs_trans,ys_trans = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2])
	xs, xs_trans = tf.meshgrid(xs,xs_trans);ys, ys_trans = tf.meshgrid(ys,ys_trans)
	Rx,Ry = tf.square(tf.subtract(xs,xs_trans)),tf.square(tf.subtract(ys,ys_trans))
	R = tf.add(Rx,Ry) 
	R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))
	ones = tf.ones([tf.multiply(Row_P_N,Column_P_N),1],tf.float32)
	ones_trans = tf.transpose(ones)
	zeros = tf.zeros([3,3],tf.float32)
	Deltas1 = tf.concat([ones, cp_s_trans, R],1)
	Deltas2 = tf.concat([ones_trans,cp_s],0)
	Deltas2 = tf.concat([zeros,Deltas2],1)
	Deltas = tf.concat([Deltas1,Deltas2],0)
	##get deltas_inv
	Deltas_inv = tf.matrix_inverse(Deltas)
	Deltas_inv = tf.expand_dims(Deltas_inv,0)
	Deltas_inv = tf.reshape(Deltas_inv,[-1])
	Deltas_inv_f = tf.tile(Deltas_inv,tf.stack([N_f]))
	Deltas_inv_f = tf.reshape(Deltas_inv_f,tf.stack([N_f,19, -1]))
	cp_trans =tf.transpose(cp,perm=[0,2,1])
	zeros_f_In = tf.zeros([N_f,3,2],tf.float32)
	cp = tf.concat([cp_trans,zeros_f_In],1)
	T = tf.transpose(tf.matmul(Deltas_inv_f,cp),[0,2,1])
	return T
	pass

def repeat(x, n_repeats):
	rep = tf.transpose(
	    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
	rep = tf.cast(rep, 'int32')
	x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
	return tf.reshape(x, [-1])
	pass

def interpolate(im, x, y, out_size):
	# constants
	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]
	channels = tf.shape(im)[3]
	x = tf.cast(x, 'float32')
	y = tf.cast(y, 'float32')
	height_f = tf.cast(height, 'float32')
	width_f = tf.cast(width, 'float32')
	out_height = out_size[0]
	out_width = out_size[1]
	zero = tf.zeros([], dtype='int32')
	max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
	max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
	# scale indices from [-1, 1] to [0, width/height]
	x = (x + 1.0)*(width_f) / 2.0
	y = (y + 1.0)*(height_f) / 2.0
	# do sampling
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)
	dim2 = width
	dim1 = width*height
	base = repeat(tf.range(num_batch)*dim1, out_height*out_width)
	base_y0 = base + y0*dim2
	base_y1 = base + y1*dim2
	idx_a = base_y0 + x0
	idx_b = base_y1 + x0
	idx_c = base_y0 + x1
	idx_d = base_y1 + x1
	# use indices to lookup pixels in the flat image and restore
	# channels dim
	im_flat = tf.reshape(im, tf.stack([-1, channels]))
	im_flat = tf.cast(im_flat, 'float32')
	Ia = tf.gather(im_flat, idx_a)
	Ib = tf.gather(im_flat, idx_b)
	Ic = tf.gather(im_flat, idx_c)
	Id = tf.gather(im_flat, idx_d)

	# and finally calculate interpolated values
	x0_f = tf.cast(x0, 'float32')
	x1_f = tf.cast(x1, 'float32')
	y0_f = tf.cast(y0, 'float32')
	y1_f = tf.cast(y1, 'float32')
	wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
	wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
	wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
	wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
	output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
	return output
	pass
def meshgrid(num_batch,height, width,Column_controlP_number,Row_controlP_number):
	x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
	y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),tf.ones(shape=tf.stack([1, width])))
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
	ones = tf.ones_like(x_t_flat) 
	grid = tf.concat([ones, x_t_flat, y_t_flat,R],0)
	return grid
if __name__ == "__main__":
	U=tf.ones([1,40,40,1])
	out_size = (40,40)
	num_batch = 1
	height = 40
	width = 40
	num_channels = 1	
	T = makeT(tf.constant([[-1.,-0.33333331,0.33333337,1.,-1.,-0.33333331,0.33333337,1.,-1.,-0.33333331,0.33333337,1.,-1.,-0.33333331,0.33333337,1.,-1.,-1.,-1.,-1., -0.33333331, -0.33333331,-0.33333331,-0.33333331,0.33333337,0.33333337,0.33333337,0.33333337,1.,1.,1.,1.]]),4,4)
	T = tf.reshape(T, (-1, 2, 19))
	T = tf.cast(T, 'float32')
	height_f = tf.cast(height, 'float32')
	width_f = tf.cast(width, 'float32')
	out_height = 40
	out_width = 40

	grid = meshgrid(1,40,40,4,4)
	grid = tf.expand_dims(grid, 0)
	grid = tf.reshape(grid, [-1])
	#print("shape grid",sess.run(tf.shape(grid)))
	grid = tf.tile(grid, tf.stack([num_batch]))
	grid = tf.reshape(grid, tf.stack([num_batch, 19, -1]))
	T_g = tf.matmul(T, grid)
	x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
	y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
	x_s_flat = tf.reshape(x_s, [-1])
	y_s_flat = tf.reshape(y_s, [-1])
	input_transformed = interpolate(U, x_s_flat, y_s_flat,out_size)
	output = tf.reshape(input_transformed, tf.stack([1, 40, 40, 1]))


