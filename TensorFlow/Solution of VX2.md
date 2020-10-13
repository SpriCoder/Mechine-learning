# 1. AVX2
1. 使用TensorFlow的时候，出现了CPU处理能力和TensorFlow的支持CPU的能力不同。
2. 以下为CPU支持AVX2优化，但是安装的2.0的tensorflow版本却不支持。

# 2. 解决方法一
1. 忽略性能:我们可以屏蔽这个异常
```py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
说明：
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error
```

# 解决方法二
1. 重新编译TensorFlow源码以兼容AVX2优化
2. <a href = "https://github.com/lakshayg/tensorflow-build">github下载网址</a>

# 参考
1. <a href = "https://blog.csdn.net/jiangsujiangjiang/article/details/89330772">I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this Tensor</a>