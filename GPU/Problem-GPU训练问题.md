GPU
---
1. 本文记录了部分和GPU配置和使用过程中的问题。
  
# 1. GPU内存不够
> “Failed to get convolution algorithm. This is probably because cuDNN failed to initialize”错误的解决办法
> TF:2.0+

1. 问题原因：图像比较大导致GPU资源消耗较大
2. 可以在程序入口通过如下方式来限制内存使用
```py
# tensorflow的版本
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# keras的版本
import tensorflow as tf
import numpy as np
import keras
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
```

3. <a href = "https://blog.csdn.net/tsyccnh/article/details/102938368">“Failed to get convolution algorithm. This is probably because cuDNN failed to initialize”错误的解决办法</a>