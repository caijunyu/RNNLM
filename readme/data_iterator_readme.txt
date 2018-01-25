1.next_random(self)
    随机生成下一批训练数据。
    （1）随机生成一个0-1之间的数。
    （2）train_buckets_scale：[0.1,0.3,0.6,0.8,1]类似这样的一个list,生成的Buckets会有n个bucket，每个Bucket都有不同个句子，一个Bucket中句子的数量越多，
         这个bucket在随机生成时占的比重越大。而把这个Bucket之前的所有句子的数量相加/所有的句子的个数就是前面这些bucket的权重。变成了如上的一个scale，
         随机生成的0-1之间的数在上面的哪个区间，就选择哪个Bucket。
    （3）调用seqModel.py的get_batch方法获得inputs, outputs, weights。