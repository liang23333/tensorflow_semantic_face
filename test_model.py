import numpy as np
import cv2
import tensorflow as tf
import model

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


img=cv2.imread('example.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=img.astype(np.float)/255.0
print(img.shape)
halfImg=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
img=img.reshape((1,128,128,3))
halfImg=halfImg.reshape((1,64,64,3))
with tf.Session() as sess:
    image=tf.placeholder("float",[1,128,128,3])
    halfimg=tf.placeholder("float",[1,64,64,3])
    feed_dict={image:img,halfimg:halfImg}

    SF=model.Semantic_face('./net_P_P_S_F.mat','./net_G_P_S_F.mat')
    with tf.name_scope("Semantic_face"):
        SF.build(image,halfimg)
    out=sess.run(SF.convG32,feed_dict=feed_dict)
    print(out.shape)
    out=out[0]
    out=im2uint8(out)
    out=cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
    cv2.imwrite('example_deblur.png',out)

