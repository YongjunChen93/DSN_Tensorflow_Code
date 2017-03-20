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
    print("cp_trans reshape",sess.run(tf.shape(cp_trans)))
    print("cp_trans ",sess.run(cp_trans))
    zeros_f_In = tf.zeros([N_f,3,2],tf.float32)
    print("zeros_f_In reshape",sess.run(tf.shape(zeros_f_In)))
    print("zeros_f_In ",sess.run(zeros_f_In))
    cp = tf.concat([cp_trans,zeros_f_In],1)
    T = tf.transpose(tf.matmul(Deltas_inv_f,cp),[0,2,1])
    return T
    pass
def repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
    #print("rep",sess.run(rep))
    rep = tf.cast(rep, 'int32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    #print("x",sess.run(x))
    return tf.reshape(x, [-1])
    pass
def interpolate(im, im_org, x, y, out_size):
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
    # TPS output coordinate
    Coordinate = base_y0 + x0
    Coordinate = tf.expand_dims(Coordinate,0)
    Temp = tf.unstack(Coordinate)
    B = tf.unstack(tf.reshape(Coordinate,[-1,len(Temp)]))
    #get batch index
    Batch_index = []
    for i in range(len(Temp)):
        Batch_index.append(i)
    Batch_index = tf.transpose(tf.expand_dims(tf.stack(Batch_index),0))
    Batch_index = tf.cast(Batch_index,tf.int32)
    Batch_index = tf.unstack(Batch_index)
    Be_insert = tf.unstack(Batch_index)

    for Batch_size in range(len(B)-1):
        for index in range(len(Be_insert), 0, -1):
            Batch_index.insert(index*(Batch_size+1), Be_insert[index-1])    
    Batch_index = tf.stack(Batch_index)
    #get corresponding image coordinate
    Coordinate = tf.reshape(Coordinate,[-1,1])
    index = tf.concat([Batch_index,Coordinate],1)
    index = tf.cast(index,tf.int64)
    #Im
    Batch = tf.cast(tf.shape(im)[0],tf.int64)
    C = tf.cast(tf.shape(im)[3],tf.int64)
    H = tf.cast(tf.shape(im)[1],tf.int64)
    print("H",sess.run(H))
    W = tf.cast(tf.shape(im)[2],tf.int64)
    print("W",sess.run(W))
    #Value from U
    im=tf.reshape(im,[-1])
    print("indices",sess.run(tf.shape(index)))
    Sparse_values=tf.SparseTensor(indices=index, values=im, dense_shape=[Batch,H*W*C])
    value_from_im=tf.sparse_tensor_to_dense(sp_input=Sparse_values,default_value=-10,validate_indices=False)
    value_from_im=tf.cast(value_from_im,'float32')
    print("value_from_im",sess.run(value_from_im))
    thred=tf.subtract(tf.ones_like(value_from_im,'float32'),8)
    print("thred",sess.run(thred))
    #Check which state is selected
    S_O_R_bool=tf.Tensor.__ge__(value_from_im,thred)
    S_O_R_value=tf.cast(S_O_R_bool,tf.float32)
    S_O_R_value=tf.subtract(tf.ones_like(S_O_R_value),S_O_R_value)
    #Value from im_org
    B_org = tf.shape(im_org)[0]
    im_org=tf.reshape(im_org,[B_org,-1])
    im_org = tf.cast(im_org,'float32')
    value_from_im_org = tf.multiply(S_O_R_value,im_org)
    #Use to offset the thred value
    equal_to_thred=tf.multiply(S_O_R_value,10)
    #Ouput value
    output = tf.add(tf.add(value_from_im,value_from_im_org),equal_to_thred)
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
    U=tf.random_uniform([1,40,40,1],minval=0,maxval=130)
    U = tf.reshape(U,[40,40,1])
    U =tf.cast(U,'uint8')
    U = tf.image.encode_png(U)
    U = tf.image.decode_png(U)
    U = tf.expand_dims(U,0)
    input_summary = tf.summary.image("input_image", U)
    U=tf.random_uniform([1,40,40,1],minval=0,maxval=130)
    U_org=tf.multiply(tf.ones([1,40,40,1]),10)
    out_size = (40,40)

    x_s_flat,y_s_flat = tf.linspace(-1.,1.,out_size[0]),tf.linspace(-1.,1.,out_size[1])
    x_s_flat,y_s_flat = tf.meshgrid(x_s_flat,y_s_flat)
    x_s_flat,y_s_flat = tf.reshape(x_s_flat,[-1]),tf.reshape(y_s_flat,[-1])
    input_transformed = interpolate(U, U_org, x_s_flat, y_s_flat,out_size)

    output = tf.reshape(input_transformed, tf.stack([1, 40, 40, 1]))
    output = tf.reshape(output,[40,40,1])
    print(sess.run(tf.shape(output)))
    output =tf.cast(output,'uint8')
    output = tf.image.encode_png(output)
    output = tf.image.decode_png(output)
    output = tf.expand_dims(output,0)
    print(sess.run(tf.shape(output)))
    sess = tf.Session()
    writer = tf.summary.FileWriter('logs')
    output_summary = tf.summary.image("output_image", output)
    merged = tf.summary.merge_all()
    summary = sess.run(merged)
    writer.add_summary(summary)
    writer.close()
    sess.close()
    pass
