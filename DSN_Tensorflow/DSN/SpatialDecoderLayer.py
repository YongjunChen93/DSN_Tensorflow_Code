import tensorflow as tf
sess = tf.Session()
#h_trans_2 = spatialdecoder(h_trans,x_tensor,h_fc_loc2,out_size, Column_controlP_number,Row_controlP_number)
def spatialdecoder(U, U_org, cp,out_size, Column_controlP_number,Row_controlP_number,name='SpatialDecoderLayer', **kwargs):
    def _makeT(cp,Column_controlP_number,Row_controlP_number):
        with tf.variable_scope('_makeT'):  
            cp = tf.reshape(cp,(-1,2,16))
            cp = tf.cast(cp,'float32')       
            N_f = tf.shape(cp)[0]       
            #c_s
            x,y = tf.linspace(-1.,1.,Column_controlP_number),tf.linspace(-1.,1.,Row_controlP_number)
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
            ones = tf.ones([tf.multiply(Row_controlP_number,Column_controlP_number),1],tf.float32)
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
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _triangle_interpolate(im,im_org, x, y, out_size):
        with tf.variable_scope('_interpolate'):
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
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            #idx_a = base_y0 + x0
            #idx_b = base_y1 + x0
            #idx_c = base_y0 + x1
            #idx_d = base_y1 + x1
            #===========================Triangle_intepolate
            # TPS output coordinate
            Coordinate = base_y0 + x0
            Coordinate = tf.expand_dims(Coordinate,0)
            Temp = tf.unstack(Coordinate)
            B = tf.unstack(tf.reshape(Coordinate,[1600,len(Temp)]))
            
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
            W = tf.cast(tf.shape(im)[2],tf.int64)
            #Value from U
            im=tf.reshape(im,[-1])
            Sparse_values=tf.SparseTensor(indices=index, values=im, dense_shape=[Batch,H*W*C])
            value_from_im=tf.sparse_tensor_to_dense(sp_input=Sparse_values,default_value=-10,validate_indices=False)
            value_from_im=tf.cast(value_from_im,'float32')
            thred=tf.subtract(tf.ones_like(value_from_im,'float32'),8)
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

    def _meshgrid(height, width,Column_controlP_number,Row_controlP_number):
        with tf.variable_scope('_meshgrid'):

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
            #R1 = tf.zeros_like(R)
            #Source coordinates
            ones = tf.ones_like(x_t_flat) 
            grid = tf.concat([ones, x_t_flat, y_t_flat,R],0)
            return grid
    def _transform(T, U, U_org,out_size,Column_controlP_number,Row_controlP_number):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(U)[0]
            height = tf.shape(U)[1]
            width = tf.shape(U)[2]
            num_channels = tf.shape(U)[3]
            T = tf.reshape(T, (-1, 2, 19))
            T = tf.cast(T, 'float32')
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width,Column_controlP_number,Row_controlP_number)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            #grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))
            grid = tf.reshape(grid, tf.stack([num_batch, 19, -1]))
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(T, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            
            output_transformed = _triangle_interpolate(U,U_org,x_s_flat,y_s_flat,out_size)
            output = output_transformed
            output = U
            
            return output

    with tf.variable_scope(name):
        T = _makeT(cp,Column_controlP_number,Row_controlP_number)
        output = _transform(T, U, U_org, out_size, Column_controlP_number, Row_controlP_number)
        return output

