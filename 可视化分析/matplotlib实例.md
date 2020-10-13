```py
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文，通过设置字段完成对于紫日的实现。
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 
#———————————————————————
labels = '财经15%', '社会30%', '体育15%','科技10%', '其它30%'
sizes = [15, 30, 15, 10, 30]
explode = (0, 0.1, 0, 0,0)#突出第2项
fig1, ax1 = plt.subplots()
pie = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=90)
patches = pie[0]
patches[0].set_hatch('.')
patches[1].set_hatch('-')
patches[2].set_hatch('+')
patches[3].set_hatch('x')
patches[4].set_hatch('o')
plt.legend(patches, labels)
ax1.axis('equal')
plt.title('新闻网站用户兴趣分析')
plt.show()
#———————————————————
import numpy as np
N = 5
inMeans = (20, 25, 30, 35, 27)
outMeans = (25, 35, 34, 20, 25)
inStd = (2, 3, 4, 1, 2)
outStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    #Bar坐标位置
width = 0.5     #Bar的宽度
p1 = plt.bar(ind, inMeans, width, yerr=inStd)
p2 = plt.bar(ind, outMeans, width,bottom=inMeans, yerr=outStd)
plt.ylabel('分值')
plt.title('不同组用户下国内外用户分值')
plt.xticks(ind, ('组1', '组2', '组3', '组4', '组5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('国内', '国外'))
plt.show()
#——————————————————————————————————————————
import matplotlib.pyplot as plt
import squarify
squarify.plot(sizes=[20,10,30,40], label=["组A(20%)", "组B(10%)", "组C(30%)", "组D(40%)"], color=["red","green","blue", "grey"], alpha=.4 )
plt.axis('off')
plt.title('不同组用户比例')
plt.show()
#———————————————————————————
# library
import numpy as np
import seaborn as sns
x=range(21,26)
y=[ [10,4,6,5,3], [12,2,7,10,1], [8,18,5,7,6],[1,8,3,5,9] ]
labels = ['组A','组B','组C','组D']
pal = sns.color_palette("Set1")
plt.stackplot(x,y, labels=labels, colors=pal, alpha=0.7 )
plt.ylabel('分值')
plt.xlabel('年龄')
plt.title('不同组用户区间分值比较')
plt.legend(loc='upper right')
plt.show()
#———————————————————————————————————————————
import seaborn as sns
import ssl
import numpy as np
import matplotlib.pyplot as plt
import random
ssl._create_default_https_context = ssl._create_unverified_context
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
np.random.seed(42)

#------------------------------------------------------------------
#生成褐色散点图
N = 40
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
#area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
plt.title("随机生成的数字散点")
plt.scatter(x, y, s=100, c='black',alpha=0.8)
plt.ylabel('Y坐标值')
plt.xlabel('X坐标值')
plt.show()
#------------------------------------------------------------------


#------------------------------------------------------------------
#生成气泡图
N = 10
x=range(21,31)
y = np.random.rand(10)
z = np.random.rand(40)
colors = np.random.rand(N)
plt.title("不同年龄下各等级的分值气泡图")
plt.scatter(x, y, c=colors, s=z*1000,alpha=0.9)
plt.ylabel('等级')
plt.xlabel('年龄')
plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
##直方图
import matplotlib.mlab as mlab
np.random.seed(19680801)
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
x = mu + sigma * np.random.randn(500)
num_bins = 60
fig, ax = plt.subplots()
n, bins, patches = ax.hist(x, num_bins, normed=1)
y = mlab.normpdf(bins, mu, sigma)
ax.plot(bins, y, '--')
ax.set_xlabel('智商IQ')
ax.set_ylabel('概率密度')
ax.set_title(r'智商分布情况直方图')
fig.tight_layout()
plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
#小组得分直方图
np.random.seed(42)
number_of_bins = 30
# An example of three data sets to compare
number_of_data_points = 600
labels = ["组1", "组2", "组3"]
data_sets = [np.random.normal(0, 1, number_of_data_points),
             np.random.normal(6, 1, number_of_data_points),
             np.random.normal(0, 4, number_of_data_points)]
# Computed quantities to aid plotting
hist_range = (np.min(data_sets), np.max(data_sets))
binned_data_sets = [
                    np.histogram(d, range=hist_range, bins=number_of_bins)[0]
                    for d in data_sets
                    ]
binned_maximums = np.max(binned_data_sets, axis=1)
x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
# The bin_edges are the same for all of the histograms
bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
heights = np.diff(bin_edges)
# Cycle through and plot each histogram
fig, ax = plt.subplots()
for x_loc, binned_data in zip(x_locations, binned_data_sets):
    lefts = x_loc - 0.5 * binned_data
    ax.barh(centers, binned_data, height=heights, left=lefts)
ax.set_xticks(x_locations)
ax.set_xticklabels(labels)
ax.set_title('小组得分直方图')
ax.set_ylabel("得分")
plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
##heatmap
# library
import seaborn as sns
import pandas as pd
import numpy as np
people=np.repeat(("组1","组2","组3","组4","组5"),6)
#feature=('周一','周二','周三','周四','周五','周六')*5
feature=('1','2','3','4','5','6')*5
value=np.random.random(30)
df=pd.DataFrame({'工作日': feature, '团队': people, 'value': value })
print(df)
df_wide=df.pivot_table( index='团队', columns='工作日', values='value' )
print(df_wide.head())
p2=sns.heatmap( df_wide ).set_title("各小组工作日表现比较")
plt.show()
#------------------------------------------------------------------

#------------------------------------------------------------------
#box
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
print(spread,center,flier_high,flier_low)
fig7, ax7 = plt.subplots()
ax7.set_title('小组分值分布情况')
ax7.boxplot(data,labels=['组1'])
plt.show()
#------------------------------------------------------------------


#------------------------------------------------------------------
# 需要进行进一步思考
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def parallel_coordinates(data_sets, style=None):
    
    dims = len(data_sets[0])
    x    = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)
    
    if style is None:
        style = ['r-']*len(data_sets)
    
    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))
    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = [(value - min_max_range[dimension][0]) /
               min_max_range[dimension][2]
               for dimension,value in enumerate(ds)]
        norm_data_sets.append(nds)
    data_sets = norm_data_sets
    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, style[dsi])
            ax.set_xlim([x[i], x[i+1]])
    
    # Set the x axis ticks
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 1)
        mn   = min_max_range[dimension][0]
        for i in range(ticks):
            v = mn + i*step
            labels.append('%4.2f' % v)
        axx.set_yticklabels(labels)
    
    
    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%4.2f' % (mn + i*step) for i in range(ticks)]
    axx.set_yticklabels(labels)
    # Stack the subplots
    plt.subplots_adjust(wspace=0)
    return plt

if __name__ == '__main__':
    import random
    base  = [0,   1,  3,   5,  0]
    scale = [1.5, 2.5, 2.0, 2., 2.]
    data = [[base[x] + random.uniform(0., 1.)*scale[x]
    for x in range(5)] for y in range(30)]
    colors = ['g'] * 30
    base  = [3,   6,  0,   1,  3]
    scale = [1.5, 2., 2.5, 2., 2.]
    data.extend([[base[x] + random.uniform(0., 1.)*scale[x]
               for x in range(5)] for y in range(30)])
    colors.extend(['b'] * 30)
             
    pp = parallel_coordinates(data, style=colors)
    pp.show()
```