import tensorflow as tf
sess = tf.Session()

def repeat(x, n_repeats):
	rep = tf.transpose(
	    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
	#print("rep",sess.run(rep))
	rep = tf.cast(rep, 'int32')
	x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
	#print("x",sess.run(x))
	return tf.reshape(x, [-1])

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
	#print("max_y",sess.run(max_y))
	max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

	# scale indices from [-1, 1] to [0, width/height]
	x = (x + 1.0)*(width_f) / 2.0
	#print("after_scale",sess.run(tf.shape(x)))
	#print("after_scale",sess.run(x))
	y = (y + 1.0)*(height_f) / 2.0

	# do sampling
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	x0 = tf.clip_by_value(x0, zero, max_x)
	print("x0_clip",sess.run(x0))
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	print("y0_clip",sess.run(y0))
	y1 = tf.clip_by_value(y1, zero, max_y)
	print("y1_clip",sess.run(y1))
	dim2 = width
	print("dim2",sess.run(dim2))
	dim1 = width*height
	print("dim1",sess.run(dim1))
	#print("range",sess.run(tf.range(num_batch)*dim1))
	base = repeat(tf.range(num_batch)*dim1, out_height*out_width)
	print("base",sess.run(tf.shape(base)))
	print("base",sess.run(base))
	base_y0 = base + y0*dim2
	print("base_y0",sess.run(base_y0))
	base_y1 = base + y1*dim2
	print("base_y1",sess.run(base_y1))
	idx_a = base_y0 + x0
	print("idx_a",sess.run(idx_a))
	idx_b = base_y1 + x0
	print("idx_b",sess.run(idx_b))
	idx_c = base_y0 + x1
	print("idx_c",sess.run(idx_c))
	idx_d = base_y1 + x1
	print("idx_d",sess.run(tf.shape(idx_d)))
	print("idx_d",sess.run(idx_d))

	# use indices to lookup pixels in the flat image and restore
	# channels dim
	im_flat = tf.reshape(im, tf.stack([-1, channels]))
	print("im_flat",sess.run(tf.shape(im_flat)))
	print("im_flat",sess.run(im_flat))
	im_flat = tf.cast(im_flat, 'float32')
	Ia = tf.gather(im_flat, idx_a)
	print("Ia",sess.run(Ia))
	Ib = tf.gather(im_flat, idx_b)
	print("Ib",sess.run(tf.shape(Ib)))
	print("Ib",sess.run(Ib))
	Ic = tf.gather(im_flat, idx_c)
	print("Ic",sess.run(Ic))
	Id = tf.gather(im_flat, idx_d)
	print("Id",sess.run(Id))

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
	print("output",sess.run(output))
	return output


#file = open('1.png', 'rb')
#data = file.read()
#image = tf.image.decode_png(data, channels=1)
#image = tf.expand_dims(image, 0)
#input_summary = tf.summary.image("input_image", image)
U=tf.multiply(tf.ones([1,40,40,1]),180)
U = tf.reshape(U,[40,40,1])
U =tf.cast(U,'uint8')
U = tf.image.encode_png(U)
U = tf.image.decode_png(U)
U = tf.expand_dims(U,0)
input_summary = tf.summary.image("input_image", U)

U=tf.multiply(tf.ones([1,40,40,1]),180)
out_size = (40,40)


x_s_flat,y_s_flat = tf.linspace(-1.,1.,out_size[0]),tf.linspace(-1.,1.,out_size[1])
x_s_flat,y_s_flat = tf.meshgrid(x_s_flat,y_s_flat)
x_s_flat,y_s_flat = tf.reshape(x_s_flat,[-1]),tf.reshape(y_s_flat,[-1])
#print("x_s_flat",sess.run(tf.shape(x_s_flat)))
#print("x_s_flat",sess.run(x_s_flat))
input_transformed = interpolate(U, x_s_flat, y_s_flat,out_size)
print(sess.run(tf.shape(input_transformed)))
output = tf.reshape(input_transformed, tf.stack([1, 40, 40, 1]))
print(sess.run(tf.shape(output)))

output = tf.reshape(output,[40,40,1])
print(sess.run(tf.shape(output)))
output =tf.cast(output,'uint8')
output = tf.image.encode_png(output)
output = tf.image.decode_png(output)
output = tf.expand_dims(output,0)
print(sess.run(tf.shape(output)))
#print(sess.run(output))
sess = tf.Session()
writer = tf.summary.FileWriter('logs')

output_summary = tf.summary.image("output_image", output)
merged = tf.summary.merge_all()
summary = sess.run(merged)
writer.add_summary(summary)
writer.close()
sess.close()
#print(sess.run(output))
