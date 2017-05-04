# encoding:utf-8
# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import time
import numpy as np  # python矩阵/数组运算

finput = open("199801_clear.txt", "r")
articles = []   # 所有的文章数组
article = ''    # 一篇文章
articles_title = ''  # 文章标题
line = finput.readline()
invalid_word = ['w', 'e', 'p', 'u', 'c', 'y']   # 去除无用词
time1 = time.time()
while line:
    if line == '\n':
        line = finput.readline()
        continue
    title = line[:15]
    line = line[23:]
    temp = ''
    while line.find('/') != -1:
        slice_pos = line.find('/')
        space_pos = line.find(' ')
        if space_pos == -1:
            if line[slice_pos+1:] not in invalid_word:
                temp = temp + line[:slice_pos] + ' '
                line = ''
                continue
        if line[slice_pos+1:space_pos] not in invalid_word:
            temp = temp + line[:slice_pos] + ' '
        line = line[space_pos + 2:]
    if title != articles_title:
        articles_title = title
        if article != '':
            articles.append(article)
        article = ''
    article += temp
    line = finput.readline()
articles.append(article)
articles_count = len(articles)
print(articles[3])
time2 = time.time()
print("预处理用时：", (time2-time1))

time3 = time.time()
# 获取tf_idf词频矩阵，矩阵第i行表示的是第i篇文章，第j列表示的是j词在第i篇文章中的词频
vector = CountVectorizer()
transformer = TfidfTransformer()
tf_idf = transformer.fit_transform(vector.fit_transform(articles))
word = vector.get_feature_names()  # 获取所有的词
print("文章数为：", len(articles))
print("总词数：", len(word))

# 计算相似度  由余弦相似度来获得两篇文章的相似度，由于每篇文章都是一个词频向量，
# 所以cosθ = a . b /|a|*|b|, |a| = sqrt(a.a)也就是tf_idf * tf_idf的转置的对角线的平方根
# 但是通过取对角线发现全是1，因此相似度就变成了a . b
# diagonal = (tf_idf * tf_idf.T).diagonal()
similarity = tf_idf * tf_idf.T

# 获得了相似度矩阵过后，找出除对角线元素之外的最大值，对角线全为1表示自己和自己最相似，
# 所以减去一个单位矩阵后再取最大值，而tf_idf是行数为文章的篇数，
# 列数是含有最大词数的某篇文章的词数，乘上转置，就是行列数为文章篇数的单位矩阵
similarity_matrix = (similarity - np.eye(articles_count)).tolist()
max_positions = np.where(similarity_matrix == np.max(similarity_matrix))    # 取得所有的具有相同最大值的位置,返回tuple类型
position = max_positions[0]  # max_position[1]是对应的行列交换的位置，去掉

# position前一半是行数，后一半是列数
half_position = int(len(position) / 2)
for i in range(half_position):
    print("第", position[i], "和第", position[i+half_position], "相似度最大为：",
          similarity_matrix[position[i]][position[i+half_position]])
    print("分别是：\n", articles[position[i]], '\n', articles[position[i+half_position]])
time4 = time.time()
print("计算相似度用时：", time4-time3)
print("总用时：", time4-time1)
