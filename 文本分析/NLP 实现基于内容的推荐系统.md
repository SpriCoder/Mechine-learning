# 1. 源码
```py
# RAKE是快速自动关键字提取库
from rake_nltk import Rake
importpandas as pd
importnumpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import

# 从外部导入数据集
CountVectorizerdf =pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')
df.head()

# 查看10位受欢迎的导演
df['Director'].value_counts()[0:10].plot('barh',figsize=[8,5],
fontsize=15, color='navy').invert_yaxis()

# 数据预处理、删除停用词、标点符号、空白，并将所有单词转换为小写
# 必须使用自然语言处理对数据进行预处理，之后会以向量化的方式将这些转换为数字，然后计算余弦相似性。
df['Key_words']= ''
r =Rake()for index, row in df.iterrows():
    r.extract_keywords_from_text(row['Plot'])
    key_words_dict_scores =r.get_word_degrees()
    row['Key_words'] =list(key_words_dict_scores.keys())
df['Genre']= df['Genre'].map(lambda x: x.split(','))
df['Actors']= df['Actors'].map(lambda x: x.split(',')[:3])
df['Director']= df['Director'].map(lambda x: x.split(','))for index, row in df.iterrows():
    row['Genre'] = [x.lower().replace(' ','')for x in row['Genre']]
    row['Actors'] = [x.lower().replace(' ','')for x in row['Actors']]
    row['Director'] = [x.lower().replace(' ','') for x in row['Director']]

# 合并列属性到新列Bag_of_words中来创建词汇表征
df['Bag_of_words']= ''
columns= ['Genre', 'Director', 'Actors', 'Key_words']for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
   
df = df[['Title','Bag_of_words']]

# 为Bag_of_words创建向量表示，并创建相似性矩阵
count= CountVectorizer()
count_matrix= count.
fit_transform(df['Bag_of_words'])cosine_sim =cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)

indices= pd.Series(df['Title'])

# 运行并测试推荐模型
def recommend(title, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).
sort_values(ascending= False)
    top_10_indices =list(score_series.iloc[1:11].index)
   
    for i in top_10_indices:
       recommended_movies.append(list(df['Title'])[i])
       
    return recommended_movies

#查看复仇者联盟的推荐
recommend('TheAvengers')
```

# 2. 一些注解