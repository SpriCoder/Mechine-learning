word embedding
---

# 1. 单词表示

## 1.1. one hot representation
1. 程序中编码单词的一个方式one hot encoding，但是这太浪费空间，而且单词之间彼此无关。

## 1.2. Distributed representation
1. 单词彼此无关不符合我们的现实情况。
2. 在语义(girl&women)、复数(word&words)、时态(buy&bought)等方面没有被考虑到

### 1.2.1. Distributed representation的目的
1. 在数据量角度，机器学习就是从有限的例子中找到合理的f

# 2. 神经网络分析
1. 对于四个词，使用独热编码则一张图，而使用Distributed representation则为一张表格。
2. word embedding就是自动学习到输入空间的映射

# 3. 参考
1. <a href = "https://zhuanlan.zhihu.com/p/27830489">YJango的word Embedding介绍</a>
2. <a href = "https://blog.csdn.net/k284213498/article/details/83474972">讲清楚embedding到底在干什么</a>