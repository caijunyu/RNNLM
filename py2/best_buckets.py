# -*- coding: utf-8 -*-
def calculate_buckets(length_array, max_length, max_buckets):
    """
      :param length_array: 每个句子长度的list,形如： [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4]
      :param max_length: 句子最长的长度
      :param max_buckets: 最大的buckets数
      :return:
      """
    # 将length_array 变成【长度】 = 个数  的字典。 形如： {1: 5, 2: 10, 3: 3, 4: 2}
    d = {}
    for length in length_array:
        if not length in d:
            d[length] = 0
        d[length] += 1
    dd = [(x,d[x]) for x in d]
    dd = sorted(dd,key = lambda x:x[0])

    # 将 dd = [(1, 5), (2, 10), (3, 3), (4, 2)]  变成  running_sum 的形式[(1, 5), (2, 15), (3, 18), (4, 20)]
    # running_sum： 将小于一长度的所有句子数量加和。
    running_sum = []
    sum = 0
    for i in range(len(dd)):
        sum += dd[i][1]
        running_sum.append((dd[i][0], sum))

    # 截取句子长度不大于max_length的句子。
    end_index = 0
    for i in xrange(len(running_sum)-1, -1, -1):
        if running_sum[i][0] <= max_length: #从running_sum 的最长的句子开始与max_length对比，一直减少到比max_length小的句子的索引。
            end_index = i + 1
            break

    print("running_sum [(length, count)] :")
    print(running_sum)
    # 对新切分好的running_sum 继续寻找最优切分点，start_num是这个running_sum最小长度还小一个长度的句子的起始个数，这样可以用ll[i][0]-start_num,就得到了第i长度的句子到这个
    # bucket起始长度的句子的个数。
    # return index so that l[:index+1] and l[index+1:]
    def best_point(array, start_num):
        max_v = 0
        index = 0
        for i in xrange(len(array)):
            v = (array[-1][0] - array[i][0]) * (array[i][1] - start_num)
            if v > max_v:
                max_v = v
                index = i
        return index,max_v

    ##将分好bucket的每一个模块添加到states（array）这个list中形成
    # states:[（array1, maxv1, id1）,
    #         （array2, maxv2, id2）,
    #         （array3, maxv3, id3]，   比如： [([(1, 5), (2, 15)], 0, 0), ([(3, 18), (4, 20)], 0, 0)]
    #  每一个模块的格式为：（array1, maxv1, id1）,其中array1: 分好的这个bucket中包含的running_sum,比如[(1, 5), (2, 15)]和[(3, 18), (4, 20)]，
    # maxv1: 在 array1这个模块中，按照id1来划分bucket，可以节省下 maxv1的计算量。
    # id1: 按照 id1划分 array1 得到 bucket。

    # 本函数的作用就是在上面的states中筛选出 mavx最大的那个 array 在states list 中的序号。
    def arg_max(states):
        max_v = -10
        index = -1
        for i in xrange(len(states)):
            if states[i][1] > max_v:# 如果v > 目前最大的v，说明这个模块细分以后节省的更多，更新使用这个模块。对这个模块的划分是通过 index来划分的。
                max_v = states[i][1]
                index = i
        return index

    if end_index <= max_buckets:
        buckets = [x[0] for x in running_sum[:end_index]]
    else:
        buckets = []
        states = [(running_sum[:end_index], 0, end_index - 1, 0)] # (array,  maxv, index, start_num)
        while len(buckets) < max_buckets:
            index = arg_max(states)
            state = states[index]   #根据 index获取states中对应的（array1, maxv1, id1，start_num）
            del states[index]

            array = state[0]
            split_index = state[2]
            buckets.append(array[split_index][0])
            # 将一个bucket再次切分成两个bucket。
            array1 = array[:split_index + 1]
            array2 = array[split_index + 1:]
            if len(array1) > 0:  #新切分的左面的bucket的start_num和旧的bucket的start_num 一样，因为左面的开始序号是一样的。
                start_num = state[3]
                index, max_v = best_point(array1,start_num)
                states.append((array1, max_v, index, start_num))
            if len(array2) > 0:  #新切分的右面的bucket的start_num 应该是将旧的bucket切分以后那个切分点的数量。
                start_num = array[split_index][1]
                index, max_v = best_point(array2, start_num)
                states.append((array2, max_v, index, start_num))
    print(sorted(buckets))
    return sorted(buckets)

def split_buckets(array, buckets):
    """
    对array根据buckets进行分组，每个items就是一个句子，对句子长度len[items]调用get_bucket_id(len[items], buckets)计算出这个句子应该属于哪个buckets，
    将其存入对应的bucket中 d[index].append(items)，并更新order中每个bucket中句子的个数order.append((index, len(d[index]) - 1))
    :param array: [[items]]
    :param buckets:
    :return:
    d : [[[items]]]　　d中有buckets个list　每个list中存放的又是句子的list。
    order: [(bucket_id, index_in_bucket)]　　　存放每个bucket对应的存放的句子的个数
    """
    order = []
    d = [[] for i in xrange(len(buckets))]
    for items in array:
        index = get_buckets_id(len(items), buckets)
        if index >= 0:
            d[index].append(items)
            order.append((index, len(d[index]) - 1))
    return d, order

#用句子的长度l与buckets对比，<=buckets[i]时说明这个句子属于这个bucket,返回这个bucket在buckets中的序号id。
def get_buckets_id(l, buckets):
    id = -1
    for i in xrange(len(buckets)):
        if l <= buckets[i]:
            id = i
            break
    return id

def main():
    import random
    a = []
    for i in range(1000):
        l = random.randint(1, 50)
        a.append(l)
    max_length = 40
    max_buckets = 4
    print(calculate_buckets(a, max_length, max_buckets))

if __name__ == "__main__":
    # main()
    length_array = [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4]
    max_length = 4
    max_buckets = 3
    calculate_buckets(length_array, max_length, max_buckets)