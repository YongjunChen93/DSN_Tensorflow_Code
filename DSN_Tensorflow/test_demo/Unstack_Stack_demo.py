import tensorflow as tf 
sess = tf.Session()
Coordinate=tf.constant([[0,1,2,1],[1,3,2,1]])
T = tf.unstack(Coordinate)
inputs = []
for i in range(len(T)):
	inputs.append(i)

inputs = tf.transpose(tf.expand_dims(tf.stack(inputs),0))
inputs = tf.cast(inputs,tf.int32)

rows = tf.unstack(inputs)
be_insert = tf.unstack(inputs)
for Batch_size in range(len(T)+1):
	for index in range(len(be_insert), 0, -1):
		rows.insert(index*(Batch_size+1), be_insert[index-1])	
rows = tf.stack(rows)
print(sess.run(rows))