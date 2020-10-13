KNN
---
最近邻算法

# 1. KNN算法思想
1. 一个样本属于哪个类别，用K个已知类别的最近邻，然后这K个里面哪个最多，这个样本属于哪一个类。

# 2. KNN算法的计算步骤
1. 对数据进行标准化，通常是进行归一化，避免量纲对计算距离的影响；
2. 计算待分类数据与训练集中每一个样本之间的距离；
3. 找出与待分类样本距离最近的k个样本；
4. 观测这k个样本的分类情况；
5. 把出现次数最多的类别作为待分类数据的类别。

# 3. KNeighborsClassifier函数
`sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=None, **kwargs)`
1. n_neighbors:临近的节点数量，默认值是5
2. weights:权重，默认值是uniform，
    + uniform：表示每个数据点的权重是相同的；
    + distance：离一个簇中心越近的点，权重越高；
    + callable：用户定义的函数，用于表示每个数据点的权重
3. algorithm
    + auto：根据值选择最合适的算法
    + ball_tree：使用BallTree
    + kd_tree：KDTree
    + brute：使用Brute-Force查找
4. leaf_size
    + leaf_size传递给BallTree或者KDTree，表示构造树的大小，用于影响模型构建的速度和树需要的内存数量，最佳值是根据数据来确定的，默认值是30。
5. p，metric，metric_paras
    + p参数用于设置Minkowski 距离的Power参数，当p=1时，等价于manhattan距离；当p=2等价于euclidean距离，当p>2时，就是Minkowski 距离。
    + metric参数：设置计算距离的方法
    + metric_paras：传递给计算距离方法的参数
6. n_jobs
    +  并发执行的job数量，用于查找邻近的数据点。默认值1，选取-1占据CPU比重会减小，但运行速度也会变慢，所有的core都会运行。

# 4. 观察数据
1. 由于KNN分类是监督式的分类方法之前，构建一个复杂的分类模型之前，首先需要已标记的数据集
```py
from sklearn.datasets import load_iris
iris_dataset=load_iris()
iris_dataset.keys()
```
2. 样本数据:data是样本数据，共4列150行，列名是由feature_names来确定的，每一列叫做矩阵的一个特征(属性)
3. 标签:target是标签，用数字表示，target_names是标签的文本表示。
4. 查看数据的散点图:
```py
import pandas as pd
import mglearn
iris_df=pd.DataFrame(x_train,columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20}
                    ,s=60,alpha=.8,cmap=mglearn.cm3)
```

# 5. 创建模型

## 5.1. 拆分数据
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
```

>sklearn.model_selection.train_test_split(*arrays, test_size ,train_size ,random_state ,shuffle ,stratify )
1. test_size：拆分的测试集数据所占的百分比，如果test_size和train_size都是none，那么默认值是test_size=0.25
2. train_size：拆分的训练集数据所占的百分比，
3. random_state：如果是int，那么参数用于指定随机数产生的种子；如果是None，使用np.random作为随机数产生器
4. shuffle：布尔值，默认值是True，表示在拆分之前对数据进行洗牌；如果shuffle = False，则分层必须为None。
5. stratify：如果不是None，则数据以分层方式拆分，使用此作为类标签。

## 5.2. 创建分类器
1. 使用KNeighborsClassifier创建分类器，设置参数n_neighbors为1：
```py
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
```

## 5.3. 使用训练集来构建模型
>knn.fit(x_train, y_train)

## 5.4. 预测新数据
```py
x_new=np.array([[5, 2.9, 1, 0.2]])
prediction= knn.predict(x_new)
print("prediction :{0}  ,classifier:{1}".format(prediction,iris_dataset["target_names"][prediction]))
```

# 6. 模型评估

## 6.1. 模型的正确率
1. 在使用模型之前，应该使用测试集来评估模型，所谓模型的正确率，就是使用已标记的数据，根据数据预测的结果和标记的结果进行比对，计算比对成功的占比：
```py
assess_model_socre=knn.score(x_test,y_test)
print('Test set score:{:2f}'.format(assess_model_socre))
```

## 6.2. 邻居个数
1. 如何确定邻居的个数，下面使用枚举法，逐个测试邻居的个数，并根据模型的score()函数查看模型的正确率。
```py
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer=load_breast_cancer()

x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]

neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    knn=KNeighborsClassifier(n_neighbors)
    knn.fit(x_train,y_train)
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label='Training Accuracy')
plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
```

## 6.3. 距离选择

## 6.4. 预测的不确定度估计

# 7. 参考
1. <a href = "https://www.cnblogs.com/ljhdo/p/10600613.html">sklearn学习</a>