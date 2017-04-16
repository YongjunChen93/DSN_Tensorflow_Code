import tensorflow as tf
import time
import datetime
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
        with tf.variable_scope('_triangle_interpolate'):
            # constants
            num_batch = im.shape[0].value
            height = im.shape[1].value
            width = im.shape[2].value
            channels = im.shape[3].value
            A_U,B_U,C_U,D_U = _split_input(im,width,height)
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
            X = tf.reshape(x,[num_batch,height,width,channels])
            Y = tf.reshape(y,[num_batch,height,width,channels])
            # get A B C D coordinate
            Ax,Bx,Cx,Dx = _split_input(X,width,height)
            Ay,By,Cy,Dy = _split_input(Y,width,height)
            # get min coordinate
            Min_X = tf.minimum(tf.minimum(Ax,Bx),tf.minimum(Cx,Dx))
            Min_Y = tf.minimum(tf.minimum(Ay,By),tf.minimum(Cy,Dy))
            #get max coordinate
            Max_X = tf.maximum(tf.maximum(Ax,Bx),tf.maximum(Cx,Dx))
            Max_Y = tf.maximum(tf.maximum(Ay,By),tf.maximum(Cy,Dy))
            Px_min = tf.cast(tf.minimum(tf.maximum(tf.ceil(Min_X),0),width-1),'int32')
            Py_min = tf.cast(tf.minimum(tf.maximum(tf.ceil(Min_Y),0),height-1),'int32')
            Px_max = tf.cast(tf.maximum(tf.minimum(tf.floor(Max_X),width-1),0),'int32')
            Py_max = tf.cast(tf.maximum(tf.minimum(tf.floor(Max_Y),width-1),0),'int32')
            PX = Px_min
            PY = Py_min
            #final Value from U
            Value_from_U_final = tf.zeros([num_batch,height*width*channels])
            Value_from_U_final = tf.cast(Value_from_U_final,'float32')
            Weights = tf.zeros([num_batch,height*width*channels])
            gap_x = tf.zeros_like(PX)
            gap_y = tf.zeros_like(PY)
            for i in range(4):
                PX = PX + gap_x
                PY = PY + gap_y
                Px = tf.cast(PX,'float32')
                Py = tf.cast(PY,'float32')
                Value_ABC,weight_ABC = _triange_function(Px,Py,Ax,Ay,Bx,By,Cx,Cy,A_U,B_U,C_U)
                Value_BCD,weight_BCD = _triange_function(Px,Py,Dx,Dy,Bx,By,Cx,Cy,D_U,B_U,C_U)
                Value_ACD,weight_ACD = _triange_function(Px,Py,Ax,Ay,Dx,Dy,Cx,Cy,A_U,D_U,C_U)
                Value_ABD,weight_ABD = _triange_function(Px,Py,Ax,Ay,Bx,By,Dx,Dy,A_U,B_U,D_U)
                Weight_all = tf.clip_by_value(weight_ABC + weight_BCD + weight_ACD + weight_ABD,0.001,1e+10)
                Value_final =  (Value_ABC+Value_BCD + Value_ACD + Value_ABD)/Weight_all
                coordx= tf.cast(tf.reshape(Px,[-1]),'int32')
                coordy = tf.cast(tf.reshape(Py,[-1]),'int32')
                dim2 = width
                dim1 = width*height
                Base = _repeat(tf.range(channels)*dim1, (out_height)*(out_width))
                Base = tf.tile(Base,tf.stack([num_batch]))
                Base = tf.reshape(Base,[num_batch,height,width,channels])
                Base,_,_,_ = _split_input(Base,height,width)
                Base = tf.reshape(Base,[-1])
                Base_Y = Base + coordy*dim2
                coordinate = Base_Y + coordx
                coordinate = tf.expand_dims(coordinate,0)
                #get batch index
                batch_index = []
                for i in range(num_batch):
                    batch_index.append(i)
                batch_index = tf.transpose(tf.expand_dims(tf.stack(batch_index),0))
                batch_index = tf.cast(batch_index,tf.int32)
                batch_index = tf.reshape(tf.tile(batch_index,[1,(height-1)*(width-1)*channels]),[-1,1])
                #get corresponding image coordinate
                coordinate = tf.reshape(coordinate,[-1,1])  
                Index = tf.concat([batch_index,coordinate],1)  
                Index = tf.cast(Index,tf.int64)
                #Im
                #Value from U
                Value_final=tf.reshape(Value_final,[num_batch*(height-1)*(width-1)*channels])
                sparse_values=tf.SparseTensor(indices=Index, values=Value_final, dense_shape=[num_batch,height*width*channels])
                Value_from_U=tf.sparse_tensor_to_dense(sp_input=sparse_values,default_value=-10,validate_indices=False)
                Value_from_U=tf.cast(Value_from_U,'float32')
                #weights
                thred=tf.subtract(tf.ones_like(Value_from_U,'float32'),8)
                weights=tf.Tensor.__ge__(Value_from_U,thred)
                weights=tf.cast(weights,tf.float32)
                Weights += weights
                Value_from_U_final+=Value_from_U
                gap_x = tf.less(0,Px_max-PX)
                gap_y = tf.less(0,Py_max-PY)
                gap_x = tf.cast(gap_x,'int32')
                gap_y = tf.cast(gap_y,'int32')
            #Weights=tf.cast(Weights,'float32')
            Value_from_U_final = Value_from_U_final/Weights
            Thred=tf.subtract(tf.ones_like(Value_from_U_final,'float32'),8)
            #Check which state is selected
            S_o_r_bool=tf.Tensor.__ge__(Value_from_U_final,Thred)
            S_o_r_value=tf.cast(S_o_r_bool,tf.float32)
            S_o_r_value=tf.subtract(tf.ones_like(S_o_r_value),S_o_r_value)
            #Value from im_org
            b_org = tf.shape(im_org)[0]
            im_org=tf.reshape(im_org,[b_org,-1])
            im_org = tf.cast(im_org,'float32')
            Value_from_im_org = tf.multiply(S_o_r_value,im_org) 
            #Use to offset the thred value
            Equal_to_thred=tf.multiply(S_o_r_value,10) 
            #Ouput value
            output = tf.add(tf.add(Value_from_U_final,Value_from_im_org),Equal_to_thred)
            return output
    def _meshgrid(U,height, width,Column_controlP_number,Row_controlP_number):
        with tf.variable_scope('_meshgrid'):
            #initial
            start = time.clock()
            num_batch = U.shape[0].value
            height = U.shape[1].value
            width = U.shape[2].value
            channels = U.shape[3].value
            #generate output coordinate
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
            grid = tf.tile(grid,tf.stack([1,channels]))
            return grid
    def _transform(T, U, U_org,input_size,out_size,Column_controlP_number,Row_controlP_number):
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
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            # 19 * (H * W * C)

            grid = _meshgrid(U, out_height, out_width,Column_controlP_number,Row_controlP_number)
            meshgrid_time = time.clock()
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            ## B * (19 * H * W * C)
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, Column_controlP_number*Row_controlP_number+3, -1]))
            T_g = tf.matmul(T, grid)
            # x = B * 1 * (H * W * C)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            output_transformed = _triangle_interpolate(U,U_org,x_s_flat,y_s_flat,input_size,out_size)
            output = tf.reshape(output_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(T, U, U_org, input_size,out_size, Column_controlP_number, Row_controlP_number)
        return output










