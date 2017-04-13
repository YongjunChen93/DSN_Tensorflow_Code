import tensorflow as tf
import time
import datetime
sess = tf.Session()
#h_trans_2 = spatialdecoder(h_trans,x_tensor,h_fc_loc2,out_size, Column_controlP_number,Row_controlP_number)
def TPS_decoder(U, U_org, T,input_size,out_size, Column_controlP_number,Row_controlP_number,name='SpatialDecoderLayer', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])
    def _triange_function(Px,Py,Ax,Ay,Bx,By,Cx,Cy,A_U,B_U,C_U):
        with tf.variable_scope('_triange_function'):
            S_ABC = tf.abs((Ax * By + Bx * Cy + Cx * Ay - Ay * Bx -By * Cx -Cy * Ax)*0.5)
            S_PBC = tf.abs((Px * By + Bx * Cy + Cx * Py - Py * Bx -By * Cx -Cy * Px)*0.5)
            S_APC = tf.abs((Ax * Py + Px * Cy + Cx * Ay - Ay * Px -Py * Cx -Cy * Ax)*0.5)
            S_ABP = tf.abs((Ax * By + Bx * Py + Px * Ay - Ay * Bx -By * Px -Py * Ax)*0.5)
            true_table=tf.less_equal(tf.abs(S_PBC+S_APC+S_ABP-S_ABC),0.00001*tf.ones_like(S_ABC,'float32'))
            true_table=tf.cast(true_table,'float32')
            Value = tf.multiply((A_U*S_PBC + B_U*S_APC + C_U*S_ABP)/S_ABC,true_table)
            return Value,true_table
    def _split_input(im,width,height):
        with tf.variable_scope('_split_input'):
            Left,_= tf.split(im,[width-1,1],2)
            _,Right= tf.split(im,[1,width-1],2)
            A,_ = tf.split(Left,[height-1,1],1)
            B,_ = tf.split(Right,[height-1,1],1)
            _,C = tf.split(Left,[1,height-1],1)
            _,D = tf.split(Right,[1,height-1],1)
            return A,B,C,D                            
    def _triangle_interpolate(im,im_org, x, y,input_size, out_size):
        print("==============triangle_interpolate start=================")
        with tf.variable_scope('_triangle_interpolate'):
            print("shape_im",im.get_shape())
            # constants
            num_batch = im.shape[0].value
            height = im.shape[1].value
            width = im.shape[2].value
            channels = im.shape[3].value
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = height
            out_width = width
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(im.shape[1].value - 1, 'int32')
            max_x = tf.cast(im.shape[2].value - 1, 'int32')
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
            base = _repeat(tf.range(channels)*dim1, out_height*out_width)
            base = tf.tile(base,tf.stack([num_batch]))
            base_y0 = base + y0*dim2

            # TPS output coordinate
            Coordinate = base_y0 + x0
            Coordinate = tf.expand_dims(Coordinate,0)           
            #get batch index
            Batch_index = []
            for i in range(num_batch):
                Batch_index.append(i)
            Batch_index = tf.transpose(tf.expand_dims(tf.stack(Batch_index),0))
            Batch_index = tf.cast(Batch_index,tf.int32)
            print("Batch_index",Batch_index.get_shape())
            Batch_index = tf.reshape(tf.tile(Batch_index,[1,height*width*channels]),[-1,1])
            #get corresponding image coordinate
            Coordinate = tf.reshape(Coordinate,[-1,1])   
            index = tf.concat([Batch_index,Coordinate],1)    
            index = tf.cast(index,tf.int64)
            #Im
            Batch = tf.cast(im.shape[0].value,tf.int64)
            C = tf.cast(im.shape[3].value,tf.int64)
            H = tf.cast(im.shape[1].value,tf.int64)
            W = tf.cast(im.shape[2].value,tf.int64)
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
            print("output_shape",output.get_shape())
            return output

    def _meshgrid(U,height, width,Column_controlP_number,Row_controlP_number):
        print("=================meshgrid start=====================")
        with tf.variable_scope('_meshgrid'):
            #initial
            start = time.clock()
            num_batch = U.shape[0].value
            height = U.shape[1].value
            width = U.shape[2].value
            channels = U.shape[3].value
            t1 = time.clock()
            print("meshgrid initial",t1-start)
            #generate output coordinate
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            px,py = tf.stack([x_t_flat],axis=2),tf.stack([y_t_flat],axis=2)
            t2 = time.clock()
            print("output coordinate",t2-t1)
            #source control points
            x,y = tf.linspace(-1.,1.,Column_controlP_number),tf.linspace(-1.,1.,Row_controlP_number)
            x,y = tf.meshgrid(x,y)
            xs,ys = tf.transpose(tf.reshape(x,(-1,1))),tf.transpose(tf.reshape(y,(-1,1)))
            cpx,cpy = tf.transpose(tf.stack([xs],axis=2),perm=[1,0,2]),tf.transpose(tf.stack([ys],axis=2),perm=[1,0,2])
            px, cpx = tf.meshgrid(px,cpx);py, cpy = tf.meshgrid(py,cpy)
            t3 = time.clock()
            print("input coordinate",t3-t2)
            #Compute distance R
            Rx,Ry = tf.square(tf.subtract(px,cpx)),tf.square(tf.subtract(py,cpy))
            R = tf.add(Rx,Ry)
            t4 = time.clock()
            print("Compute distance R",t4-t3)
            #cp = tf.transpose(cp)
            R = tf.multiply(R,tf.log(tf.clip_by_value(R,1e-10,1e+10)))

            #R1 = tf.zeros_like(R)
            #Source coordinates
            ones = tf.ones_like(x_t_flat) 
            grid = tf.concat([ones, x_t_flat, y_t_flat,R],0)
            grid = tf.tile(grid,tf.stack([1,channels]))
            t5 = time.clock()
            print("get grid",t5-t4)
            print("=================meshgrid end=====================")
            return grid
    def _transform(T, U, U_org,input_size,out_size,Column_controlP_number,Row_controlP_number):
        print("======================transform start===================")
        with tf.variable_scope('_transform'):
            start= time.clock()
            num_batch = U.shape[0].value
            height = U.shape[1].value
            width = U.shape[2].value
            num_channels = U.shape[3].value
            T = tf.reshape(T, (-1, 2, Column_controlP_number*Row_controlP_number+3))
            T = tf.cast(T, 'float32')

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = height
            out_width = width
            t1 = time.clock()
            print("transform_inital",t1-start)
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            # 19 * (H * W * C)

            grid = _meshgrid(U, out_height, out_width,Column_controlP_number,Row_controlP_number)
            meshgrid_time = time.clock()
            print("total meshgrid time",meshgrid_time-t1)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            ## B * (19 * H * W * C)
            grid = tf.tile(grid, tf.stack([num_batch]))
            ## B * 19 * (H * W * C)
            grid = tf.reshape(grid, tf.stack([num_batch, Column_controlP_number*Row_controlP_number+3, -1]))
            t2 = time.clock()
            print("grid",t2-meshgrid_time)
            # T = B * 2 * 19
            # grid = B * 19 * ( H * W * C )
            T_g = tf.matmul(T, grid)
            t3 = time.clock()
            print("T_matmul",t3-t2)
            # x = B * 1 * (H * W * C)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            t4 = time.clock()
            output_transformed = _triangle_interpolate(U,U_org,x_s_flat,y_s_flat,input_size,out_size)
            t5 = time.clock()
            print("total output transformed",t5-t4)
            output = tf.reshape(output_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            t6= time.clock()
            print("output reshape",t6-t5)
            print("======================transform end======================")
            return output

    with tf.variable_scope(name):
        print("=================Decoder start===================")
        start = time.clock()
        print("T shape",T.get_shape())
        T_time = time.clock()
        print("total_makeT_time",T_time-start)
        output = _transform(T, U, U_org, input_size,out_size, Column_controlP_number, Row_controlP_number)
        tranform_time = time.clock()
        print("transform_time",tranform_time - T_time)
        print("=================Decoder end===================")
        return output










