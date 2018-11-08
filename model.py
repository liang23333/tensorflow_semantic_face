import tensorflow as tf
import numpy as np
from scipy.io import loadmat

class Semantic_face:
    def __init__(self,P_model_path=None,G_model_path=None):
        if P_model_path is not None:
            self.P_mat=loadmat(P_model_path)['net_P']
            print('P_mat loaded success')
        else:
            print('P_mat loaded failure ,exit')
            exit()
        if G_model_path is not None:
            self.G_mat=loadmat(G_model_path)['net_G']
            print('G_mat loaded success')
        else:
            print('G_mat loaded failure ,exit')
            exit()
    def conv_layer(self,bottom,id,name):
        filter=tf.constant(self.P_mat[0,id]['w'])
        bias=tf.constant(self.P_mat[0,id]['b'][:,0])
        conv=tf.nn.conv2d(bottom,filter,[1,1,1,1],padding='SAME',name=name)
        conv_bias=tf.nn.bias_add(conv,bias)
        return conv_bias
    def bn_layer(self,bottom,id,name):
        mean,var=tf.nn.moments(
            bottom,
            axes=[0,1,2]
        )
        scale=tf.constant(self.P_mat[0,id]['bw'][:,0])
        shift=tf.constant(self.P_mat[0,id]['bb'][:,0])
        epsilon=1e-4
        bn=tf.nn.batch_normalization(bottom,mean,var,shift, scale,epsilon,name=name)
        return bn
    def leaky_relu(self,bottom,alpha,name):
        return tf.nn.leaky_relu(bottom,alpha=alpha,name=name)
    def maxpool(self,bottom,name):
        return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

    def conv_transpose(self,bottom,id,name):
        N=bottom.get_shape()[0]
        H=bottom.get_shape()[1]
        W=bottom.get_shape()[2]
        C=bottom.get_shape()[3]
        shape=tf.TensorShape([N,H+H,W+W,C])
        filter=tf.constant(self.P_mat[0,id]['upw'])
        bias=tf.constant(self.P_mat[0,id]['upb'][:,0])
        conv_t=tf.nn.conv2d_transpose(bottom,filter,output_shape=shape,strides=[1,2,2,1],name=name)
        conv_t_bias=tf.nn.bias_add(conv_t,bias)
        return conv_t_bias

    def conv_bn_relu(self,bottom,id,name):
        conv = self.conv_layer(bottom, id, name='conv' + name + '_' + str(id))
        bn = self.bn_layer(conv, id, name='bn' + name + '_' + str(id))
        relu = self.leaky_relu(bn, alpha=0.0, name='relu' + name + '_' + str(id))
        return relu


    def conv_bn_relu_pool(self,bottom,id,name):
        relu=self.conv_bn_relu(bottom,id,name)
        pool=self.maxpool(relu,name='maxpool'+name+'_'+str(id))
        return relu,pool


    def tran_conv_bn_relu(self,bottom,id,down_val,name):
        tran=self.conv_transpose(bottom,id,'conv_transpose'+name+'_'+str(id))
        conv = self.conv_layer(tran, id, name='conv' + name + '_' + str(id))
        bn = self.bn_layer(conv, id, name='bn' + name + '_' + str(id))
        relu = self.leaky_relu(bn, alpha=0.0, name='relu' + name + '_' + str(id))
        return relu


    def tran_conv_bn_relu_sum(self,bottom,id,down_val,name):
        relu = self.tran_conv_bn_relu(bottom,id,down_val,name)
        return down_val+relu


    def convG_layer(self,bottom,id,name):
        filter = tf.constant(self.G_mat[0, id]['w'],dtype=tf.float32)
        bias = tf.constant(self.G_mat[0, id]['b'][:,0],dtype=tf.float32)
        conv = tf.nn.conv2d(bottom, filter, [1, 1, 1, 1], padding='SAME', name=name)
        conv_bias = tf.nn.bias_add(conv, bias)
        return conv_bias

    def convG_relu(self,bottom,id,name):
        convG=self.convG_layer(bottom,id,name='conv'+name+'_'+str(id))
        relu=self.leaky_relu(convG,alpha=0.0,name='relu'+name+'_'+str(id))
        return relu

    def convG_transpose(self,bottom,id,name):
        N=bottom.get_shape()[0]
        H=bottom.get_shape()[1]
        W=bottom.get_shape()[2]
        C=bottom.get_shape()[3]
        shape=tf.TensorShape([N,H+H,W+W,C])
        filter=tf.constant(self.G_mat[0,id]['w'])
        bias=tf.constant(self.G_mat[0,id]['b'][:,0])
        conv_t=tf.nn.conv2d_transpose(bottom,filter,output_shape=shape,strides=[1,2,2,1],name=name)
        conv_t_bias=tf.nn.bias_add(conv_t,bias)
        return conv_t_bias


    def ResBlockG(self,bottom,id1,id2,name):
        conv1=self.convG_relu(bottom,id1,name=name)
        conv2=self.convG_layer(conv1,id2,name=name)
        res=bottom+conv2
        relu=self.leaky_relu(res,alpha=0.0,name='resBlock'+str(id1))
        return relu

    def build(self,blur,halfImg):
        ### Face Parsing

        #### downsample
        self.conv_bn_reluP_1,self.pool_P_1=self.conv_bn_relu_pool(blur,0,name='P')
        self.conv_bn_reluP_2,self.pool_P_2=self.conv_bn_relu_pool(self.pool_P_1,1,name='P')
        self.conv_bn_reluP_3,self.pool_P_3=self.conv_bn_relu_pool(self.pool_P_2,2,name='P')
        self.conv_bn_reluP_4,self.pool_P_4=self.conv_bn_relu_pool(self.pool_P_3,3,name='P')
        self.conv_bn_reluP_5,self.pool_P_5 = self.conv_bn_relu_pool(self.pool_P_4, 4, name='P')

        self.conv_bn_reluP_6=self.conv_bn_relu(self.pool_P_5,5,name='P')


        #### upsample
        self.tran_conv_bn_relu_sum_7 = self.tran_conv_bn_relu_sum(self.conv_bn_reluP_6, 6, self.conv_bn_reluP_5,
                                                                  name='P')
        self.tran_conv_bn_relu_sum_8 = self.tran_conv_bn_relu_sum(self.tran_conv_bn_relu_sum_7, 7, self.conv_bn_reluP_4,
                                                                  name='P')
        self.tran_conv_bn_relu_sum_9 = self.tran_conv_bn_relu_sum(self.tran_conv_bn_relu_sum_8, 8, self.conv_bn_reluP_3,
                                                                  name='P')
        self.tran_conv_bn_relu_sum_10 = self.tran_conv_bn_relu_sum(self.tran_conv_bn_relu_sum_9, 9, self.conv_bn_reluP_2,
                                                                  name='P')
        self.tran_conv_bn_relu_sum_11 = self.tran_conv_bn_relu(self.tran_conv_bn_relu_sum_10, 10, self.conv_bn_reluP_1,
                                                                  name='P')
        self.conv12=self.conv_layer(self.tran_conv_bn_relu_sum_11,11,name='convP_12')


        ### net_G


        self.faceLabel=tf.nn.softmax(self.conv12,name='face_label')
        self.halfFaceLabel=self.maxpool(self.faceLabel,name='half_face_label')
        self.halfBlurImage=halfImg
        self.G_input=tf.concat([self.halfBlurImage,self.halfFaceLabel],axis=-1)


        #### scale1



        self.convG_relu1=self.convG_relu(self.G_input,0,name='G')

        ''' there is a big error with origin result'''
        self.convG_relu2=self.convG_relu(self.convG_relu1,1,name='G')
        ''' there is a big error with origin result'''


        self.convG_relu3=self.convG_relu(self.convG_relu2,2,name='G')
        self.res3=self.ResBlockG(self.convG_relu3,3,4,name='G')
        self.res5=self.ResBlockG(self.res3,5,6,name='G')
        self.res7=self.ResBlockG(self.res5,7,8,name='G')
        self.res9=self.ResBlockG(self.res7,9,10,name='G')
        self.res11=self.ResBlockG(self.res9,11,12,name='G')
        self.convG_relu13=self.convG_relu(self.res11,13,name='G')
        self.convG_relu14=self.convG_relu(self.convG_relu13,14,name='G')
        self.convG15=self.convG_layer(self.convG_relu14,15,name='conG15')


        self.convG_t16=self.convG_transpose(self.convG15, 16, name='scale1_out')
        self.G2_input=tf.concat([self.convG_t16,blur,self.faceLabel],axis=-1)

        #### scale2

        self.convG_relu18 = self.convG_relu(self.G2_input, 17, name='G')
        self.convG_relu19 = self.convG_relu(self.convG_relu18, 18, name='G')
        self.convG_relu20 = self.convG_relu(self.convG_relu19, 19, name='G')
        self.res20 = self.ResBlockG(self.convG_relu20, 20, 21, name='G')
        self.res22 = self.ResBlockG(self.res20, 22, 23, name='G')
        self.res24 = self.ResBlockG(self.res22, 24, 25, name='G')
        self.res26 = self.ResBlockG(self.res24, 26, 27, name='G')
        self.res28 = self.ResBlockG(self.res26, 28, 29, name='G')
        self.convG_relu30 = self.convG_relu(self.res28, 30, name='G')
        self.convG_relu31 = self.convG_relu(self.convG_relu30, 31, name='G')
        self.convG32 = self.convG_layer(self.convG_relu31, 32, name='conG32')




