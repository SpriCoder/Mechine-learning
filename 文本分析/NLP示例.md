```py
import nltk
import math
import string
#from nltk.corpus import stopwords
#from collections import Counter
#from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
vectorizer = TfidfVectorizer(min_df=1)
cret = vectorizer.fit_transform(corpus)
print(cret)
fnames = vectorizer.get_feature_names()
print(fnames)
arr = vectorizer.fit_transform(corpus).toarray()
print(arr)
```
```py
import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import jieba
import jieba.analyse
 
train = []
stopwords = codecs.open('stopwords.txt','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
fp = codecs.open('news.txt','r',encoding='utf8')
for line in fp:
    line = list(jieba.cut(line))
    #print(line)
    train.append([ w for w in line if w not in stopwords ])
dictionary = corpora.Dictionary(train)
corpus = [ dictionary.doc2bow(text) for text in train ]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
 lda.print_topics(5)
lda.print_topic(2)
```