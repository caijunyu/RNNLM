best_bucket.py:
   算法：贪心、分治
   主要思想：反复切分最好的那个bucket，直到bucket的数量达到max_buckets为止。选取最好的bucket是通过寻找切分这个bucket会节省最大的计算量来寻找的即max_v最大的那个bucket。
   　　　　　
   1.输入是一个每个句子的长度的list:[1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4]，句子的最大长度 max_length, bucket的个数： max_buckets.
   2.词典d, 首先统计每个句子长度分别有多少个句子，形成字典d:{1: 5, 2: 10, 3: 3, 4: 2}
   3.dd list, 将字典d变成一个list,并根据句子的长度进行排序,dd = [(1, 5), (2, 10), (3, 3), (4, 2)]
   4.running_sum, 将dd list 变成running_sum,就是统计句子长度小于某一长度的所有的值：[(1, 5), (2, 15), (3, 18), (4, 20)]
   5.根据max_length的要求，在running_sum中返回<=max_length的句子长度在running_sum中的序号 end_index
   6.如果end_index < max_buckets,那么直接把每个句子长度当做一个bucket就可以了。
   7.如果end_index > max_buckets：
    （1）states:
        设置states List 存放的是states: [([(1, 5), (2, 15)], 5, 0, 0), ([(3, 18), (4, 20)], 3, 0, 15)]的数据，
        其中每一个元素是（array1, maxv1, id1, start_num1）的形式,其中array1: 分好的这个bucket中包含的running_sum,比如[(1, 5), (2, 15)]和[(3, 18), (4, 20)]，
        maxv1: 在 array1这个模块中，按照id1来划分bucket，可以节省下 maxv1的计算量。id1: 按照 id1划分 array1 得到 bucket。
        start_num是这个running_sum最小长度还小一个长度的句子的起始个数，这样可以用ll[i][0]-start_num,就得到了第i长度的句子到这个。
    （2）index = arg_max(states)
        因为states中存储了每一个bucket再继续进行细分时的index 以及 max_v, 那么只需要遍历states中的每一个元素，找出max_v最大的那个bucket，
        返回对应的index便于后续用 state = states[index]将这个bucket的内容取出为后续处理做准备.
    （3）在原states中将取出来的bucket删除掉 del states[index]，后续将这个bucket再次切分以后再append到states中。
    （4）将state 按照 state中的 split_index = state[3]进行切分，得到新的array1 和 array2。并将split_index对应的句子的长度加入到buckets中作为一个bucket的节点。
    （5）对array1 和 array2分别计算最佳的切分点并返回在这个切分点上节省的计算量。index, max_v = best_point(array1,start_num)
        def best_point(array1,start_num):
            对于一个切分点，节省的计算量是（该buckets最长的句子长度 - 切分点的句子长度） × （buckets最长的句子长度的个数 - 切分点句子长度的个数）
            切分点句子长度的个数 = 小于等于切分点句子长度的个数 - start_num
            start_num 是小于切分点句子长度的个数
            返回 对这个 bucket 的最佳切分点的index 以及对应的节省的计算量 max_v
        注意：array1的start_num和原array的start_num是一样的，即start_num = state[3],因为是切分的左面部分。
        　　　array2的start_num应该是切分处的数量，即start_num = array[split_index][1]。因为是切分的右面部分。起始个数应该是切分点处。
        最后将处理过的array1 和 array2 按照（array1, maxv1, id1, start_num1）的格式append 到 states中。
    （６）重复上面的步骤，反复切分每一个bucket，直到bucket的数量满足需求即可。

split_buckets(array, buckets)：
   对所有的句子根据buckets分配到对应的list中，最后返回的是[list1,list2,list3]，list1中存放的是符合buckets[0]的长度的句子[s1,s2,s3,s4],s1是句子对应的编码id[1,15,21,88]

get_bucket_id(l, buckets):
　　用句子的长度l与buckets对比，<=buckets[i]时说明这个句子属于这个bucket,返回这个bucket在buckets中的序号id。
