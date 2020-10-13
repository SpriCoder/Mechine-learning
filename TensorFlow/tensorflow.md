tensorflow
---
1. 在机器学习中，我们经常需要优化器来求解最小损失。

# 1. tensorflow的安装

## 1.1. windows版本下的安装tensorflow
1. 在windows操作系统下安装最新版的tensorflow
    + cpu版使用:`pip install tensorflow`
    + gpu版使用:`pip install tensorflow-gpu`
2. 安装指定版本的tensorflow
    + `pip install tensorflow==1.14.0(版本号)`

## 1.2. linux系统下安装tensorflow

### 1.2.1. 小问题
1. 找不apt-get命令包？因为目前执行的操作系统是CentOS，在CentOS中，yum和apt-get其大致相同的作用。<a href = "https://blog.csdn.net/Mercuriooo/article/details/89070149">更多</a>

## 1.3. TensorFlow使用Docker方式安装
1. 拉取tensorflow的docker影像:`docker pull tensorflow/tensorflow`
2. 创建TensorFlow的容器，并且要启用8888端口:
```
docker run --name tf01 -d -p 8888:8888 -p 6006:6006 tensorflow/tensorflow
# 2019年4月12日更新：新的版本改用西门的命令
docker run -d  --name tf01 --rm -v /root/tf01/notebooks:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-py3-jupyter
```
3. 取得token:`docker exec -it tf01 bash`
4. 之后获得链接地址:`http://localhost:8888/?token=84f79011f319994673985cb8d8cf08160108e93c3b16ebb0 :: /notebooks`(例如)

### 1.3.1. 参考
1. <a href = "https://blog.csdn.net/zhangchao19890805/article/details/78781003">docker安装tensorflow</a>

# 2. tensorflow中的优化器

## 2.1. 常见的优化器
1. GradientDescentOptimizer
2. AdagradOptimizer
3. AdagradDAOptimizer
4. MomentumOptimizer
5. AdamOptimizer
6. RMSPropOptimizer

## 2.2. 随机梯度下降SGD

## 2.3. Adam算法
1. Adam也是梯度下降的一种，每次迭代参数的学习率有一定范围，不会因为梯度很大而导致学习率(步长)变得很大，参数的值相对比较稳定。

## 2.4. 参考
1. <a href = "https://blog.csdn.net/lomodays207/article/details/84027365">tensorflow中的优化器</a>
2. <a href = "https://www.jianshu.com/p/410bac3bf41f">AdamOptimizer</a>

# 3. tensorflow中的部分函数详解

## 3.1. tf.train.shuffle_batch函数解析
>tf.train.shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)
2. 作用:读取一个文件并且加载一个张量中的batch_size行
3. <a href = "https://blog.csdn.net/u013555719/article/details/77679964">tf.train.shuffle_batch函数详解</a>

## 3.2. concat函数的用法
1. 标注的用法:`tf.concat(values,concat_dim,name='concat')`
    + `concat_dim`:表示从哪个维度上进行连接，0基
```py
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["file3.csv", "file4.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[1], [1], [1], [1]]
col1, col2, col3, col4 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.concat(0, [[col1], [col2], [col3], [col4]])
#features = tf.stack([col1, col2, col3, col4]) #把上一行注释掉用这一行也可以


init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print(sess.run([features]))
#   for i in range(9):
#       print(sess.run([features]))

    coord.request_stop()
    coord.join(threads)
```

## 3.3. 打开Session
```py
with tf.compat.v1.Session() as sess
```

## 3.4. tf.constant
1. tf.constant用来创建常量
```py
tf.constant(
    value,
    dtype = None,
    shape = None,
    name = 'Const',
    verify_shape = False
)
```
2. 内容可以是一个值，也可以是一个列表
3. 填充方法:`tf.constant(-1,[2,3])`
    + 如果列表与shape大小，则会用列表的最后一项元素填充剩余的张量元素
4. <a href = "https://blog.csdn.net/csdn_jiayu/article/details/82155224">TensorFlow创建常量(tf.constant)详解</a>

# 4. 外部训练集的导入

## 4.1. csv文件的导入
```py
import tensorflow as tf
filename_queue = tf.train.string_input_producer(["file3.csv", "file4.csv"])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1]]
col1, col2, col3, col4 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.concat(0, [col1, col2, col3, col4])
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #for i in range(1200):
    # Retrieve a single instance:
    #example, label = sess.run([features])
    print(sess.run([features]))
    coord.request_stop()
    coord.join(threads)
```

# 5. 参考
1. <a href = "https://blog.csdn.net/umbrellalalalala/article/details/81091175">tensorflow读取csv入门</a>