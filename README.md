# RNNLM
运行于 Python  2.5
       tensorflow 1.2

1.train 数据集有42068条数据
2.valid 数据 3370条  dev_ppx:86.6203138058
3.test 数据 3370条

数据概览：
   1.多少行句子：
       wc train
   2.多少个单词（token）
       wc train
   3.多少种单词（type）
      cat train|tr '' '\n'|sort\uniq\wc
   4.最长（最短）句子有多少单词
     cat train|tr ' ' '\n'|sort|uniq -c|sort -n
     sort: 词本身排序
     uniq -c:去重记次数
     sort -n:按次数排序
    5.句子长度与数量的关系
        awk '{print NF}'|sort -n|uniq -c

4.data 文件夹
    存储原始数据集

5.data_util.py
    数据预处理
    1.创建词典
    2.根据字典将原始数据转换为对应的数字id。

6.generate_jobs.py
   生成  random research  200个参数的 bash
