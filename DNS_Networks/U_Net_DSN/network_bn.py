import tensorflow as tf
#from spatial_transformer import Affine_transformer
from TPS_transformer import TPS_transformer
from SpatialDecoderLayer import TPS_decoder
from ops import *
from Data_generator import *

class Unet(object):
    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.conv_kernel_size = (3,3)
        self.pool_kernel_size = (2,2)
        self.tps_out_size = (40,40)
        self.Column_controlP_number = 4
        self.Row_controlP_number = 4
        self.inserttps = 3
        self.input_size_D = (40, 40)
        self.out_size_D = (40, 40)
        self.Column_controlP_number_D = 4
        self.Row_controlP_number_D = 4
        if conf.use_gpu:
            self.data_format = 'NCHW'
            self.axis, self.channel_axis = (2,3),1
            self.input_shape = [self.conf.batch, self.conf.channel, self.conf.height, self.conf.width]
            self.test_input_shape = [self.conf.test_batch,self.conf.channel,1024,1024]
            self.output_shape = [self.conf.batch, self.conf.class_num, self.conf.height, self.conf.width]
            self.test_output_shape = [self.conf.test_batch,self.conf.channel,1024,1024]
        else:
            self.data_format = 'NHWC'
            self.axis, self.channel_axis = (1,2),3
            self.input_shape = [self.conf.batch, self.conf.height, self.conf.width, self.conf.channel]
            self.test_input_shape = [self.conf.test_batch,1024,1024,self.conf.channel]
            self.output_shape = [self.conf.batch, self.conf.height, self.conf.width, self.conf.class_num]
            self.test_output_shape = [self.conf.test_batch,1024,1024,self.conf.class_num]
        self.inputs = tf.placeholder(tf.float32, self.input_shape, 'inputs')
        self.test_inputs = tf.placeholder(tf.float32, self.test_input_shape,'test_inputs')
        self.label = tf.placeholder(tf.float32,self.output_shape, 'label')
        self.test_label = tf.placeholder(tf.float32,self.test_output_shape, 'test_outputs')
        self.build_network()
        #self.build_test()
        self.train_acc = self.get_accuracy(self.label,self.train_predict,'train')
        #self.test_acc = self.get_accuracy(tlabel = self.test_label, plabel = self.test_predict, scope='test')
        #self.test_acc_summary = tf.summary.scalar('test_acc', self.test_acc)
        self.train_acc_summary = tf.summary.scalar('train_acc', self.train_acc)
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)

    def get_accuracy(self,tlabel, plabel, scope):
        correct = tf.equal(tf.argmax(tlabel,self.channel_axis), tf.argmax(plabel,self.channel_axis), name = scope+'/correct')
        return tf.reduce_mean(tf.cast(correct,tf.float32), name = scope + '/accuracy')

    def build_network(self):
        outputs = self.inputs
        down_outputs = []
        cp_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            if layer_index == self.inserttps:
                outputs = self.construct_down_block(outputs, name, down_outputs, cp_outputs, first=is_first,TPS = True)
            else:
                outputs = self.construct_down_block(outputs, name, down_outputs, cp_outputs, first=is_first,TPS = False)
            print("down ",layer_index," shape ", outputs.get_shape())
        outputs = self.construct_bottom_block(outputs, 'bottom')
        print("bottom shape",outputs.get_shape())
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index==0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            if layer_index == self.inserttps:
                cp = cp_outputs[0]
                Decoder = True
            else:
                Decoder = False
                cp = []
            outputs = self.construct_up_block(outputs, down_inputs, name, cp, final=is_final, Decoder = Decoder)
            print("up ",layer_index," shape ",outputs.get_shape())
        self.train_predict = outputs
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.label, logits = outputs))
        self.loss_summary = tf.summary.scalar('loss', self.loss_op)

    def construct_down_block(self, inputs, name, down_outputs, cp_outputs, first=False,TPS=False):
        num_outputs = self.conf.start_channel_num if first else 2*inputs.shape[self.channel_axis].value
        conv1 = conv2d(inputs, num_outputs,self.conv_kernel_size,name+'/conv1',self.data_format)
        if TPS == True:
            #pass
            conv1, cp = TPS_transformer(conv1,conv1,self.tps_out_size,self.Column_controlP_number,self.Row_controlP_number)
            cp_outputs.append(cp)
            #print(conv1.get_shape())
        conv2 = conv2d(conv1, num_outputs, self.conv_kernel_size,name+'/conv2',self.data_format)
        down_outputs.append(conv2)
        pool = pool2d(conv2,self.pool_kernel_size,name+'/pool',self.data_format)
        return pool

    def construct_bottom_block(self, inputs, name):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = conv2d(inputs, 2*num_outputs, self.conv_kernel_size, name+'/conv1', self.data_format)
        

        conv2 = conv2d(conv1, num_outputs,self.conv_kernel_size,name+'/conv2',self.data_format)
        return conv2

    def construct_up_block(self, inputs, down_inputs, name, cp, final = False,Decoder=False):
        num_outputs = inputs.shape[self.channel_axis].value
        conv1 = deconv2d(inputs,num_outputs,self.conv_kernel_size,name+'/conv1',self.data_format)
        conv1 = tf.concat([conv1, down_inputs],self.channel_axis,name=name+'/concat')
        if Decoder == True:
            conv1 = TPS_decoder(conv1,conv1,cp,self.input_size_D,self.out_size_D, self.Column_controlP_number_D,self.Row_controlP_number_D)
        conv2 = conv2d(conv1,num_outputs,self.conv_kernel_size,name+'/conv2',self.data_format)
        num_outputs = self.conf.class_num if final else num_outputs/2
        conv3 = conv2d(conv2, num_outputs, self.conv_kernel_size, name+'/conv3',self.data_format)
        return conv3
    
    def build_test(self):
        outputs = self.test_inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            if layer_index == self.insertaffine:
                outputs = self.construct_down_block(outputs, name, down_outputs, first=is_first,Affine = True)
            else:
                outputs = self.construct_down_block(outputs, name, down_outputs, first=is_first,Affine = False)
        outputs = self.construct_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index==0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.construct_up_block(outputs, down_inputs, name, final=is_final)
        self.test_predict = outputs

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        data_generator = Data_generator([self.conf.batch, self.conf.height, self.conf.width],self.data_format)
        train_generator = data_generator.train_generator()
        test_generator = data_generator.valid_generator(self.conf.test_batch)
        train_step = tf.train.AdamOptimizer(self.conf.learning_rate).minimize(self.loss_op)
        self.merged_train = tf.summary.merge([self.train_acc_summary,self.loss_summary])
        #self.merged_test = tf.summary.merge([self.test_acc_summary])
        self.train_writer = tf.summary.FileWriter(self.conf.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.conf.log_dir + '/test')
        self.sess.run(tf.global_variables_initializer())
        for iter_num in range(1,self.conf.max_epoch+1):
            image,label = next(train_generator)
            summary, loss,  acc_train, _ = self.sess.run([self.merged_train, self.loss_op, self.train_acc, train_step],feed_dict={self.inputs: image,self.label: label})
            self.train_writer.add_summary(summary, iter_num)
            print("Epoch: [%2d],   loss = %.8f ,  accuracy = %.8f " %(iter_num,loss,acc_train))
            if iter_num % self.conf.save_step == 0:
                self.save(iter_num)
            if iter_num % self.conf.test_step == 0:
                pass
         #       self.test(self.conf.batch,test_generator,iter_num)
        self.train_writer.close()
        self.test_writer.close()

    def test(self,test_num,test_generator,step):
        test_img,test_label = next(test_generator)
        summary,_,test_acc = self.sess.run([self.merged_test,self.test_predict,self.test_acc],feed_dict={self.test_inputs:test_img, self.test_label:test_label})
        print("Epoch: [%2d], test_acc = %.8f" % (step,test_acc))
        self.test_writer.add_summary(summary,step)

    def save(self,step):
        print('---->saving')
        print (self.conf.log_dir)
        checkpoint_path = os.path.join(self.conf.log_dir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self,step):
        checkpoint_path = os.path.join(conf.log_dir, conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path):
            print('------- no such checkpoint')
            return
        self.saver.restore(self.sess, model_path)




