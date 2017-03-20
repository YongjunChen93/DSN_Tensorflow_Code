import tensorflow as tf
sess = tf.Session()
def repeat(x, n_repeats):
	rep = tf.transpose(
	    tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
	rep = tf.cast(rep, 'int32')
	x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
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
def meshgrid(height, width):

    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([ones,x_t_flat, y_t_flat],0)
    return grid


U=tf.ones([1,40,40,1])
print(sess.run(U))
out_size = (40,40)
num_batch = 1
height = 28
width = 28
num_channels = 1
theta = tf.constant([[0, 1., 0], [0, 0, 1.]])
theta = tf.reshape(theta, (-1, 2, 3))
theta = tf.cast(theta, 'float32')
print(sess.run(tf.shape(theta)))

height_f = tf.cast(height, 'float32')
width_f = tf.cast(width, 'float32')
out_height = 40
out_width = 40

grid = meshgrid(40,40)
grid = tf.expand_dims(grid, 0)
grid = tf.reshape(grid, [-1])
grid = tf.tile(grid, tf.stack([1]))
grid = tf.reshape(grid, tf.stack([1, 3, -1]))
print("grid",sess.run(grid))
print("theta",sess.run(theta))
T_g = tf.matmul(theta, grid)
print(sess.run(tf.shape(T_g)))
print(sess.run(T_g))
x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
print(sess.run(tf.shape(x_s)))
x_s_flat = tf.reshape(x_s, [-1])
y_s_flat = tf.reshape(y_s, [-1])
#print(sess.run(x_s_flat))
#print(sess.run(y_s_flat))
input_transformed = interpolate(U, x_s_flat, y_s_flat,out_size)
print(sess.run(input_transformed))
output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
print(sess.run(tf.shape(output)))
print(sess.run(output))


