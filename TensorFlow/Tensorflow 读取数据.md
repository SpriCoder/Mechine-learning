tensorflow读取数据
---

# 1. preloaded data
1. 由tensorflow产生数据，然后运行
```py
import tensorflow as tf
sess = tf.Session()
#设计计算图及产生数据
x1 = tf.constant([1,2,3])
x2 = tf.constant([4,5,6])
y = tf.add(x1,x2)
#session计算
print (sess.run(y))
```

# 2. feeding
1. 有python生成数据，然后由placeholder喂给tensorflow
2. placeholder:占位符
```py
#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import tensorflow as tf
import numpy as np
# 定义placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# 定义乘法运算
output = tf.multiply(input1, input2)
# 通过session执行乘法运行
with tf.Session() as sess:
    # 执行时要传入placeholder的值
    print sess.run(output, feed_dict = {input1:[7.], input2: [2.]})
```

# 3. reading from file
1. 方法一:在图里面由tensorflow产生数据并计算，当数据量大的时候会出现图传输错误的问题。
2. 方法二:才用占位符的时候，数据量过大也会是很大的传输开销。

## 3.1. tf.train.batch
>tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
1. example, label: 表示样本和样本标签，这个可以是一个样本和一个样本标签 
2. batch_size:返回的一个batch样本集的样本个数 
3. capacity:队列的长度 
4. 函数返回值都是一个按照输入顺序的batch的样本和样本标签。
5. 单个Reader读取，但是可以多线程。

# 4. tf.train.shuffle_batch
>tf.train.shuffle_batch([example, label], batch_size=batch_size, capacity=capacity, min_after_dequeue)
1. min_after_dequeue:是出队后，队列至少剩下的数据个数，小于capacity参数的值，否则会出错。也就是说这个函数的输出结果是一个乱序的样本排列的batch，不是按照顺序排列的
2. 返回值都是一个随机的batch的样本及其对应的样本标签。 
3. 单个Reader读取，但是可以多线程

# 5. tf.train.batch_join
1. 多个Reader读取，每个Reader一个线程

# 6. tf.train.shuffle_bathc_join
1. 多个Reader读取，每个Reader一个线程

# 7. 单个读取，单个样本
```py
# 按顺序输出 使用 batch
import tensorflow as tf
#生成一个先入先出队列和一个Queuerunner，生成文件名队列
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
#定义 reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义 decoder
example, label = tf.decode_csv(value, record_defaults=[['null'],['null']])#['null']解析为string类型 ，[1]为整型，[1.0]解析为浮点。
example_batch, label_batch = tf.train.batch([example, label], batch_size=1, capacity=200, num_threads=2)#保证样本和标签一一对应
#运行图
with tf.Session() as sess:
    coord = tf.train.Coordinator()#创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队
    for i in range(10):
        e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)
    coord.request_stop()
    coord.join(threads)

# 随机输出 使用shuffle_batch
import tensorflow as tf
#生成一个先入先出队列和一个Queuerunner，生成文件名队列
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义 decoder
example, label = tf.decode_csv(value, record_defaults=[['null'],['null']])
example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=1, capacity=200,min_after_dequeue=100, num_threads=2)
#运行图
with tf.Session() as sess:
    coord = tf.train.Coordinator()#创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队
    for i in range(10):
        e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)
    coord.request_stop()
    coord.join(threads)
```

# 8. 多个读取，多个样本
```py
# 按顺序输出 batch
import tensorflow as tf
#生成一个先入先出队列和一个Queuerunner，生成文件名队列
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义 decoder
record_defaults = [['null'], ['null']]  
#定义了多种解码器,每个解码器跟一个reader相连  
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]  # Reader设置为2  
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
example_batch, label_batch = tf.train.batch_join(example_list, batch_size=5) 
#运行图
with tf.Session() as sess:
    coord = tf.train.Coordinator()#创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队
    for i in range(10):
        e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)
    coord.request_stop()
    coord.join(threads)
# 随机顺序输出 shuffle_batch
import tensorflow as tf
#生成一个先入先出队列和一个Queuerunner，生成文件名队列
filenames = ['A.csv', 'B.csv', 'C.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
#定义reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义 decoder
record_defaults = [['null'], ['null']]  
#定义了多种解码器,每个解码器跟一个reader相连  
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(3)]  # Reader设置为3  
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。  
example_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=5,capacity=200,min_after_dequeue=100) 
#运行图
with tf.Session() as sess:
    coord = tf.train.Coordinator()#创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner，此时文件名队列已经进队
    for i in range(10):
        e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)
    coord.request_stop()
    coord.join(threads)
```

# 9. 迭代控制
1. 设置epoch参数，指定我们的样本在训练的时候只能被用多少轮
```py
import tensorflow as tf  
filenames = ['A.csv', 'B.csv', 'C.csv']  
#num_epoch: 设置迭代数  
filename_queue = tf.train.string_input_producer(filenames, shuffle=False,num_epochs=3)  
reader = tf.TextLineReader()  
key, value = reader.read(filename_queue)  
record_defaults = [['null'], ['null']]  
#定义了多种解码器,每个解码器跟一个reader相连  
example_list = [tf.decode_csv(value, record_defaults=record_defaults) for _ in range(2)]  # Reader设置为2  
# 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。  
example_batch, label_batch = tf.train.batch_join(example_list, batch_size=1)  
#初始化本地变量  
init_local_op = tf.initialize_local_variables()#如果不初始化，运行就会报错
with tf.Session() as sess:  
    sess.run(init_local_op)  #初始化
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord)  
    try:  
        while not coord.should_stop():  
            e_val, l_val = sess.run([example_batch, label_batch])
        print (e_val, l_val)

    except tf.errors.OutOfRangeError:  
            print('Epochs Complete!')  
    finally:    
        coord.request_stop()  
    coord.join(threads)  
    coord.request_stop()  
    coord.join(threads) 
```

# 10. 使用队列读取csv
```py
import tensorflow as tf  
# 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
filenames = ['D.csv']  
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  
# 定义Reader  
reader = tf.TextLineReader()  
key, value = reader.read(filename_queue)  
# 定义Decoder  
record_defaults = [[1], [1], [1], [1], [1]] #解析为整数
col1, col2, col3, col4, col5 = tf.decode_csv(value,record_defaults=record_defaults)  
features = tf.stack([col1, col2, col3])#前3列数据，后2列标签  
label = tf.stack([col4,col5])  
example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)  
# 运行Graph  
with tf.Session() as sess:  
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)  
    for i in range(10):  
        e_val,l_val = sess.run([example_batch, label_batch])  
        print (e_val,l_val)  
    coord.request_stop()  
    coord.join(threads) 
```

# 11. 参考
1. <a href = "https://blog.csdn.net/xuan_zizizi/article/details/78400839">tensorflow读取数据</a>