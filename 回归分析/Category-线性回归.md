Category-线性回归
---
1. 通常我们会将分类问题转化为回归问题
2. 实现方法
   1. 引入指示变量，简单地标示每一个数据点所属的类别，问题转化为用数据来预测指示变量
   2. 进行重复的回归，每次对其中的一个类别
3. 如何执行回归
   1. 最简单：尽量使每个数据点与我们拟合的直线之间的距离最小。
   2. 常见的最小化方法：最小二乘优化

$$
\sum\limits_{j=0}\limits^{N}(t_j - \sum\limits_{i = 0}\limits^{M}\beta_ix_{ij})^2 \Leftrightarrow (\bold{t} - \bold{X}\beta)^T(\bold{t}-\bold{X}\beta)
$$

- 符号说明
  - $\bold{t}$是目标值
  - $\bold{X}$是输入值矩阵(包含偏置输入)
- 求解最小值：对$\beta$求导，并将导数设置为0，矩阵求导的方法：$\frac{\delta x^Ta}{\delta x} = \frac{\delta a^Tx}{\delta x} = a$

$$
\begin{aligned}
   \frac{\delta (\bold{t} - \bold{X}\beta)^T(\bold{t}-\bold{X}\beta)}{\delta \beta} = 0\\
   \frac{\delta(t^Tt - t\bold{X}\beta - \beta^T\bold{X}^Tt + \beta^T\bold{X}^T\bold{X}\beta)}{\delta \beta} = 0\\
   -t\bold{X} - \bold{X}^Tt + \bold{X}^T\bold{X}\beta + \beta^T\bold{X}^T\bold{X} = 0 \\
   2 * \bold{X}^T\bold{X}\beta - 2 * \bold{X}t = 0 \\
   \bold{X}^T\bold{X}\beta - \bold{X}t = 0 \\
   \bold{X}^T\bold{X}\beta = \bold{X}t \\
   \beta = (\bold{X}^T\bold{X})^{-1}\bold{X}t \\
\end{aligned}
$$

- 那么对于输入向量$\bold{z}$，我们的右侧值为$\bold{z}\beta$

# 1. 线性回归的例子
课本90页