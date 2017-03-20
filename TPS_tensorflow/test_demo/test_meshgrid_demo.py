import tensorflow as tf
sess = tf.Session()
def meshgrid(height, width):
		with tf.variable_scope('meshgrid'):
			# This should be equivalent to:
			#  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
			#                         np.linspace(-1, 1, height))
			#  ones = np.ones(np.prod(x_t.shape))
			#  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
			x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
							tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
			y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
							tf.ones(shape=tf.stack([1, width])))

			x_t_flat = tf.reshape(x_t, (1, -1))
			y_t_flat = tf.reshape(y_t, (1, -1))
            
			ones = tf.ones_like(x_t_flat)
			grid = tf.concat([x_t_flat, y_t_flat, ones],0)
			return grid
theta = tf.constant([[[1,2,3],[4,5,6]]])
theta = tf.cast(theta, 'float32')
print(sess.run(tf.shape(theta)))

grid = meshgrid(28, 28)
grid = tf.expand_dims(grid, 0)
grid = tf.reshape(grid, [-1])
grid = tf.tile(grid, tf.stack([1]))
grid = tf.reshape(grid, tf.stack([1, 3, -1]))
T_g = tf.matmul(theta, grid)
x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
x_s_flat = tf.reshape(x_s, [-1])
y_s_flat = tf.reshape(y_s, [-1])
print(sess.run(tf.shape(x_s_flat)))





