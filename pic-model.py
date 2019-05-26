"""
this script takes in color photos from a pictures dir
and converts them to 3D models of the things in the photos(4 per model)
Preparation:
1: replace paths at top to respective paths in your system
(data_stuff is my dir that houses the phong clone, 
rendered images, run_seg2102 and the generated .off for training(not the saved benchmark ones)
)
2: lines: 243 and 304 replace 2 with amount of sets of images(4 images per set)
"""


import tensorflow as tf
from open3d import *
import os
import PIL.Image as Image

bat_path = "E:/PycharmProjects/3DMesh_Development/data_stuff/run_seg2102" # where bat file is
epoch_save_path = "E:/PycharmProjects/3DMesh_Development/data_stuff/idk_epoch.off" # where to save off file to 
lsave_path = "E:/PycharmProjects/3DMesh_Development/PHOTOS_GAN_SAVES/" # dir for epoch benchmarks
xtrain = np.array([])
picpath = "E:/PycharmProjects/3DMesh_Development/pictures/" # dir with pictures
npicpath = "E:/PycharmProjects/3DMesh_Development/data_stuff/tmp/idk_epoch" # dir for renderings
for x in os.listdir(picpath):
    ia = Image.open(picpath + x)
    ia = np.array(ia.convert('L'))
    print(ia.shape)
    xtrain = np.append(xtrain, ia)

print(xtrain.shape)
vertex_amt = [1000]

sess = tf.Session()


dw_1 = tf.get_variable(name="dw_1", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
db_1 = tf.get_variable(name="db_1", shape=[816, 612], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
dw_2 = tf.get_variable(name="dw_2", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
db_2 = tf.get_variable(name="db_2", shape=[102, 153], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
dw_3 = tf.get_variable(name="dw_3", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
db_3 = tf.get_variable(name="db_3", shape=[51, 51], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
dw_4 = tf.get_variable(name="dw_4", shape=[2601, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       # start shape changing via matmul
                       trainable=True, dtype=tf.float32)
db_4 = tf.get_variable(name="db_4", shape=[1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)


# start of 3D GEN variables

gw_1 = tf.get_variable(name="gw_1", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_1 = tf.get_variable(name="gb_1", shape=[1, 3264, 1224, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gw_2 = tf.get_variable(name="gw_2", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_2 = tf.get_variable(name="gb_2", shape=[1, 816, 306, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gw_3 = tf.get_variable(name="gw_3", shape=[2, 2, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_3 = tf.get_variable(name="gb_3", shape=[1, 816, 102, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32) # know shape
gw_4 = tf.get_variable(name="gw_4", shape=[1, 816, 102, 3], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_4 = tf.get_variable(name="gb_4", shape=[1, 816, 3, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32) # know
gw_5 = tf.get_variable(name="gw_5", shape=[1, 3, 816, 1000], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_5 = tf.get_variable(name="gb_5", shape=[3, 1000, 1, 1], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gw_6 = tf.get_variable(name="gw_6", shape=[1, 3, 1000, 1333], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)
gb_6 = tf.get_variable(name="gb_6", shape=[1333, 3], initializer=tf.initializers.random_normal(stddev=0.02),
                       trainable=True, dtype=tf.float32)

sess.run(tf.global_variables_initializer())


def generator(image):
    # do calculations
    # problem  Incompatible shapes between op input and calculated input gradient not with biases
    print("gen prints start:")
    g = tf.nn.conv2d(image, gw_1, strides=[1, 2, 4, 1], padding="SAME")
    g = g + gb_1
    g = tf.nn.leaky_relu(g)
    g = tf.reshape(g, [1, 3264, 1224, 1])
    g = tf.nn.conv2d(g, gw_2, strides=[1, 4, 4, 1], padding="SAME")
    print(g.shape)
    g = g + gb_2
    g = tf.nn.leaky_relu(g)
    g = tf.nn.conv2d(g, gw_3, strides=[1, 1, 3, 1], padding="SAME")
    g = g + gb_3
    g = tf.nn.relu(g)
    print(g.shape)
    g = tf.reshape(g, shape=[1, 816, 1, 102]) # ---debugged up to here---
    g = tf.matmul(g, gw_4)
    print("m4:")
    print(g.shape)
    g = tf.reshape(g, shape=[1, 816, 3, 1])
    g = g + gb_4
    g = tf.nn.relu(g)
    print(g.shape)
    g = tf.reshape(g, shape=[1, 3, 1, 816])
    g = tf.reshape(tf.matmul(g, gw_5), shape=[3, 1000, 1, 1]) + gb_5
    g = tf.nn.softmax(g)
    print(g.shape)
    g = tf.reshape(g, shape=[1, 3, 1, 1000])
    g = tf.matmul(g, gw_6)
    g = tf.reshape(g, shape=[1333, 3])
    g = g + gb_6
    g = tf.nn.softmax(g)
    print("g end")
    print(g.shape)
    return g


def discriminator(data):
    # do calculations
    print("descriminator stars here:")
    d = tf.nn.conv2d(data, filter=dw_1, strides=[1, 8, 8, 1], padding="SAME")
    print(d.shape)
    d = tf.reshape(d, shape=[816, 612])
    d = d + db_1
    d = tf.nn.relu(d)
    d = tf.reshape(d, shape=[1, 816, 612, 1])
    print(d.shape)
    d = tf.nn.conv2d(d, filter=dw_2, strides=[1, 8, 4, 1], padding="SAME")
    d = tf.reshape(d, shape=[102, 153])
    d = d + db_2
    d = tf.nn.relu(d)
    d = tf.reshape(d, shape=[1, 102, 153, 1])
    print(d.shape)
    d = tf.nn.conv2d(d, filter=dw_3, strides=[1, 2, 3, 1], padding="SAME")
    d = tf.reshape(d, shape=[51, 51])
    d = d + db_3
    d = tf.nn.relu(d)
    print(d.shape)
    d = tf.reshape(d, shape=[1, 2601])
    d = tf.matmul(d, dw_4)
    d = tf.add(d, db_4)
    d = tf.nn.relu(d)
    d = tf.reshape(d, shape=[1])
    print(d.shape)
    print("d end")

    return d


def save_func(xdata, path):
    nls = ""
    predarr = ""
    print("seg starts")
    print(xdata.shape)
    filx = open(path, "w+", encoding="ASCII")
    for lll in range(1001):
        item = xdata[lll]
        str_arr = np.array_str(item)
        nls = str_arr.replace("[", "").replace("]", "")
        predarr = predarr + nls
        predarr = predarr + "\n"
    for iii in range(334):
        item = xdata[iii]
        item = np.multiply(np.around(item, decimals=2), 100)
        str_arr = np.array_str(item)
        for ii in range(str_arr.count(".") + 1):
            str_arr = str_arr.replace(".", " ")
        predarr = predarr + "3 " + str_arr.replace("[", "").replace("]", "")
        predarr = predarr + "\n"
    prepd = "OFF\n" + "1000 333 0\n" + predarr
    for iiii in range(prepd.count("  ") + 1):
        prepd.replace("  ", " ")
    for iiiii in range(prepd.count("  ") + 1):
        prepd.replace("  ", " ")
    filx.write(prepd)
    filx.close()
    filxr = open(path, "r+")
    for xix in filxr.readlines():
        if "OFF" or "1000 333" in xix:
            continue
        lsit_nxix = list(xix)
        lsit_nxix.remove(" ")
        str_nxix = str(lsit_nxix)
        nls = nls + str_nxix + "\n"
    filxr.write(nls)
    filxr.close()


def segment(xdata):
    nxtrain = np.array([])
    nls = ""
    predarr = ""
    print("seg starts")
    print(xdata.shape)
    filx = open(epoch_save_path, "w", encoding="ASCII")
    for lll in range(1001):
        item = xdata[lll]
        str_arr = np.array_str(item)
        nls = str_arr.replace("[", "").replace("]", "")
        predarr = predarr + nls
        predarr = predarr + "\n"
    for iii in range(334):
        item = xdata[iii]
        item = np.multiply(np.around(item, decimals=2), 100)
        str_arr = np.array_str(item)
        for ii in range(str_arr.count(".") + 1):
            str_arr = str_arr.replace(".", " ")
        predarr = predarr + "3 " + str_arr.replace("[", "").replace("]", "")
        predarr = predarr + "\n"
    prepd = "OFF\n" + "1000 333 0\n" + predarr
    for iiii in range(prepd.count("  ") + 1):
        prepd.replace("  ", " ")
    for iiiii in range(prepd.count("  ") + 1):
        prepd.replace("  ", " ")
    filx.write(prepd)
    filx.close()
    filxr = open(epoch_save_path, "r+")
    for xix in filxr.readlines():
        if "OFF" or "1000 333" in xix:
            continue
        lsit_nxix = list(xix)
        lsit_nxix.remove(" ")
        str_nxix = str(lsit_nxix)
        nls = nls + str_nxix + "\n"
    filxr.write(nls)
    filxr.close()
    os.system(bat_path)
    for xx in os.listdir(npicpath):
        nia = Image.open(npicpath + "/" + xx)
        #nia = crop(nia, []) # neeeded dims=(2448, 3264)
        nia = np.array(nia.convert('L'))
        print(nia.shape)
        nxtrain = np.append(nxtrain, nia)
    nxtrain.reshape([1, 6528, 4896, 1])

    return tf.reshape(tf.convert_to_tensor(nxtrain, dtype=tf.float32), shape=[1, 6528, 4896, 1])


# temparary data values that we can swap easily.
x_placeholder = tf.placeholder(tf.float32, shape=[1, 6528, 4896, 1], name='x_placeholder') # 2d image tf placeholder
xtrain = xtrain.reshape([2, 1, 6528, 4896, 1]) # 2, 13056, 9792

print(xtrain.shape)


def pro_pcd_arr(g):
    lg = list(sess.run(g, {x_placeholder: xtrain[1]}).tolist())
    for k in range(lg.count([0, 0, 0])):
        lg.remove([0, 0, 0])
    np.array(lg)
    return lg


# descrim should be 2D convs and GZ outputs segmented
Dx = discriminator(x_placeholder) # 2d input image against seg model
# 😃
Gz = pro_pcd_arr(generator(x_placeholder)) # reg gan makes 3d model from 2d pic
print("line 167")
Dg = discriminator(segment(np.array(Gz))) # seg model vs 2d real image

print("worked")
# defines loss functions for models

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([1], shape=[1], dtype=tf.float32),
                                                                logits=Dg))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([1], shape=[1],
                                                                                        dtype=tf.float32), logits=Dg))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant([0], shape=[1],
                                                                                        dtype=tf.float32), logits=Dx))
d_loss = d_loss_real + d_loss_real

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd' and "_" in var.name]
g_vars = [var for var in t_vars if 'g' and "_" in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

    d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars, colocate_gradients_with_ops=True)

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

d_on_generated = tf.reduce_mean(discriminator(segment(np.array(pro_pcd_arr(generator(x_placeholder))))))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

# initializes all variables with tensorflow
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

d_fake_count = 0
g_count = 0
d_real_count = 0
# define loss vars
gLoss = 0
dLossFake, dLossReal = 1, 1

# training loop
for i in range(2):
    real_image_batch = xtrain[i]

    # Train discriminator on generated images
    _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                              {x_placeholder: real_image_batch})
    d_fake_count += 1

    # Train the generator
    sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
             {x_placeholder: real_image_batch})
    g_count += 1
    # train d on real images
    sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
             {x_placeholder: real_image_batch})
    d_real_count += 1
    d_real_count, d_fake_count, g_count = 0, 0, 0

    if i % 5 == 0:

        print("TRAINING STEP", i)
        print("Descriminator_loss:" + str(dLossReal))
        #mdsaver
        sess.run(save_func(np.array(pro_pcd_arr(generator(x_placeholder))),
                           path= lsave_path + "save" + str(i) + ".off"))

    if i % 20 == 0:
        save_path = saver.save(sess, "models/pretrained_3ddcgan.ckpt", global_step=i)
        print("saved to %s" % save_path)

