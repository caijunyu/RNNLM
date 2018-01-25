1.get_batch(self, data_set, bucket_id, start_id=None)
    被data_iterator.py 的 next_random 调用，生成训练所需要的 inputs, outputs, weights。
    （1）根据 bucket_id 在buckets中获取这个bucket的句子的最长长度，既：length = self.buckets[bucket_id]
    （2）随机的在bucket_data中选取一句话，一共循环选取batch_size句。word_seq = random.choice(data_set[bucket_id])
    （3）对选出的这句话做处理， 输入去掉eos，输出去掉go。
             word_input_seq = word_seq[:-1]  # without _EOS
             word_output_seq = word_seq[1:]  # without _GO
    （4）根据输出确定weights,对输入输出加pad。
    （5）将上述处理过的batch_size个数据加入对应的3个List中。input_ids.append(word_input_seq)、output_ids.append(word_output_seq)、weights.append(target_weight)
    （6）将上面的3个list变成矩阵的形式，batch_size * bucekt的句子长度
    （7）返回inputs, outputs, weights。

 2.def __init__（）
    （1）初始化参数
    （2）配置一些参数some parameters
            包括dropoutRate、learning_rate、learning_rate_decay_op、global_step
    （3）input layer
            将输入的每个batch的句子通过embedding 转换成向量。
            定义 input 和  input_embed 以及  input_embedding（在训练的过程中会一直更新）
            建立最大长度的inputs,每次处理一个输入单词，所以需要循环最长的句子的单词个数，把所有的输入单词转换成embedding再存入Input_embed中。
    （4）LSTM
            建立lstm模型，如果是多层的就建立多层模型，加入dropout，对输入和输出都要分别加。
            每层LSTM的参数是不同的。
    （5）Output Layer
            定义  Output 和 weights
            循环添加到上面两个list中
    （6）Model with buckets
         返回self.losses[]
         对于多 buckets 我们需要对于每个 buckets 都需要计算 loss 和 update 操作。
         每个bucket分别建立一个模型，然后一个输入分别输入到这些模型中，最终的loss是说有这些loss的组合。
         1.init_state  初始化   h 和 c  均这是为0
         2.softmax_loss_function : 对输出  hts先做softmax 再做交叉熵。
         3. for j, bucket in enumerate(buckets):
               循环每一个bucket，分别建立模型
               如果j>0,设置参数共享。
               1. 通过 static_rnn  计算输入  Hts
               2. 通过 hts 计算 Logits   Logits = w* hts + b
               3. 调用sequence_loss_by_example函数生成每一个bucket的loss。
     （7）train
          调用 self.losses[]更新参数。
         如果不是forward_only:
             backward, update

3.sequence_loss_by_example（）
    调用softmax_loss_function 计算  Logits和target的交叉熵，得到 loss，对一句话中的每个Loss求和得到最终的loss。

4.step()
    实现 forward+ backward + weights_updates


