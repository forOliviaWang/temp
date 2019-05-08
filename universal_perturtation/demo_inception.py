import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time

if sys.version_info[0] >= 3:    #获取python版本信息
    from urllib.request import urlretrieve    #python3使用urllib库请求网络，urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None)将URL表示的网络对象复制到本地文件
else:
    from urllib import urlretrieve


from universal_pert import universal_perturbation
device = '/gpu:0'
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(     #tf.while_loop(cond, loop_body, init_state)，返回循环结束时的循环状态 
        lambda j,_: j < n,           #cond是一个函数，负责判断继续执行循环的条件
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),  #loop_body是每个循环体内执行的操作，负责对循环状态并行更新
        loop_vars)                   #init_state为循环的起始状态，可以包含多个tensor或TensorArray
    return jacobian.stack()   #将jacobian（TensorArray）中的元素叠起来当作一个Tensor输出

if __name__ == '__main__':

    # Parse arguments
    argv = sys.argv[1:]  #从程序外部获取参数

    # Default values
    path_train_imagenet = '/datasets2/ILSVRC2012/train'
    path_test_image = 'data/test_img.png'
    
    try:
        opts, args = getopt.getopt(argv,"i:t:",["test_image=","training_path="])  #解析命令行参数，短格式开关选项'i','t',长格式选项'test_image','training_path',以上各带一个参数。getopt函数返回两个列表：opts和args。opts为分析出的格式信息。args为不属于格式信息的剩余的命令行参数,即不是按照getopt(）里面定义的长或短选项字符和附加参数以外的信息。
    except getopt.GetoptError:
        print ('python ' + sys.argv[0] + ' -i <test image> -t <imagenet training path>')
        sys.exit(2)      #sys.exit()函数是通过抛出异常的方式来终止进程的，一般在主程序中使用此退出

    for opt, arg in opts:
        if opt == '-t':           #将-t后面的参数传给path_train_imagenet
            path_train_imagenet = arg
        if opt == '-i':     #将-i后面的参数传给path_test_image
            path_test_image = arg

    with tf.device(device):    #指定tensorflow运行设备，这里指定session在第一块GPU上运行
        persisted_sess = tf.Session()
        inception_model_path = os.path.join('data', 'tensorflow_inception_graph.pb')  #路径拼接函数，拼接结果'data\tensorflow_inception_graph.pb'

        if os.path.isfile(inception_model_path) == 0:  #os.path.isfile()传入的必须是绝对路径，用于判断某一对象是否为文件，True说明是文件
            print('Downloading Inception model...')     #如果没有data\tensorflow_inception_graph.pb，则下载inception模型
            urlretrieve ("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip", os.path.join('data', 'inception5h.zip')) #下载该url表示的网络对象到本地，文件名为data/inception5h.zip
            # Unzipping the file   
            zip_ref = zipfile.ZipFile(os.path.join('data', 'inception5h.zip'), 'r')  #zipfile中的zipFile类用来创建和读取zip文件，zipfile.zipFile用来创建一个ZipFile对象，表示一个zip文件，'r'表示打开zip文件的方式
            zip_ref.extract('tensorflow_inception_graph.pb', 'data')   #ZipFile.extract(member[, path[, pwd]])，将zip文档内的指定文件解压到当前目录。参数member指定要解压的文件名称或对应的ZipInfo对象；参数path指定了解析文件保存的文件夹；
            zip_ref.close()

        model = os.path.join(inception_model_path)

        # Load the Inception model      #读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
        with gfile.FastGFile(model, 'rb') as f:  #gfile.FastGFile(filename,mode)获取文本操作句柄,类似open().与gfile.GFile的区别在于无阻塞，会无阻塞以较快的方式获得文本操作句柄，'rb'以二进制方式读取
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')   #以上为从pb文件中调用graph的过程

        persisted_sess.graph.get_operations()   #返回graph中操作的列表

        persisted_input = persisted_sess.graph.get_tensor_by_name("input:0")  #get_tensor_by_name(name)返回给定name的tensor，input为节点名称。而<input>:0是tensor名称，表示节点的第一个输出张量
        persisted_output = persisted_sess.graph.get_tensor_by_name("softmax2_pre_activation:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 224, 224, 3))}) #执行persisted_output中的操作，计算persisted_output中的张量值，返回persisted_output的执行结果
		                                                             #feed_dict以字典形式赋值，将np.reshape()的结果赋值给persisted_input。np.reshape(a, newshape, order='C').a是需要reshape的数据，newshape是新格式。order默认为C，表示按索引数据读取元素
																	 #reshape中的-1参数表示根据其他参数的维度算出该参数

        file_perturbation = os.path.join('data', 'universal.npy')    #生成的universal扰动的文件为data/universal.npy,.npy是以未压缩的原始二进制格式保存数组
        if os.path.isfile(file_perturbation) == 0:

            # TODO: Optimize this construction part!
            print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)  #.squeeze,删除单维条目

            # Load/Create data
            datafile = os.path.join('data', 'imagenet_data.npy')
            if os.path.isfile(datafile) == 0:   #如果data下没有imagenet_data.npy
                print('>> Creating pre-processed imagenet data...')
                X = create_imagenet_npy(path_train_imagenet)   #则调用prepare_imagenet_data.py中的creat_imagenet_npy来处理位于'/datasets2/ILSVRC2012/train'处的原始图片

                print('>> Saving the pre-processed imagenet data')
                if not os.path.exists('data'):  #判断是否存在data
                    os.makedirs('data')     #如果不存在，则创建目录。os.mkdir()创建路径中的最后一级目录，如果之前目录不存在则报错。os.makedirs()递归创建多层目录

                # Save the pre-processed images
                # Caution: This can take take a lot of space. Comment this part to discard saving.
                np.save(os.path.join('data', 'imagenet_data.npy'), X)

            else:
                print('>> Pre-processed imagenet data detected')
                X = np.load(datafile)   #np.load()用于读取npy格式文件

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.2,num_classes=num_classes)   #利用universal_per.py中的universal_perturbation得到universal perturbation

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v) #将universal perturbation保存到data/universal.npy

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on an image')

        # Test the perturbation on the image
        labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n') #split('\n')按行分隔

        image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
        label_original = np.argmax(f(image_original), axis=1).flatten()   #np.argmax(a)取出a中元素最大值所对应的索引，axis=1表示按行,flatten()折成一维数组  f()如前定义，为模型输出
        str_label_original = labels[np.int(label_original)-1].split(',')[0]  ##？

        # Clip the perturbation to make sure images fit in uint8
        clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)

        image_perturbed = image_original + clipped_v[None, :, :, :]
        label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
        str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]

        # Show original and perturbed image
        plt.figure()   #创建自定义图像Figure
        plt.subplot(1, 2, 1)   #创建子图，subplot(a,b,c),a为子图的行数，b为改行的列数，c表示每行的第几个图像
        plt.imshow(undo_image_avg(image_original[0, :, :, :]).astype(dtype='uint8'), interpolation=None)  #plt.imshow()将一个image显示在二维坐标轴上
        plt.title(str_label_original)  #设置坐标轴标签

        plt.subplot(1, 2, 2)
        plt.imshow(undo_image_avg(image_perturbed[0, :, :, :]).astype(dtype='uint8'), interpolation=None)
        plt.title(str_label_perturbed)

        plt.show()  #打开matplotlib查看器，并显示绘制图形
        plt.savefig('result.svg')