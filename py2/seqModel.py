# -*- coding: utf-8 -*-
import random
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


class SeqModel(object):
    def __init__(self,
                 buckets,
                 size,  # h和c的维度
                 vocab_size,
                 num_layers,
                 max_gradient_norm,  # clip gradient
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 withAdagrad=True,
                 forward_only=False,
                 dropoutRate=1.0,
                 devices="",
                 run_options=None,
                 run_metadata=None,
                 topk_n=30,
                 dtype=tf.float32, ):
        """Create the model.
              Args:
              buckets: a list of pairs (I, O), where I specifies maximum input length
              that will be processed in that bucket, and O specifies maximum output
              length. Training instances that have inputs longer than I or outputs
              longer than O will be pushed to the next bucket and padded accordingly.
              We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
              size: number of units in each layer of the model.
              num_layers: number of layers in the model.
              max_gradient_norm: gradients will be clipped to maximally this norm.
              batch_size: the size of the batches used during training;
              the model construction is independent of batch_size, so it can be
              changed after initialization if this is convenient, e.g., for decoding.

              learning_rate: learning rate to start with.
              learning_rate_decay_factor: decay learning rate by this much when needed.

              forward_only: if set, we do not construct the backward pass in the model.
              dtype: the data type to use to store internal variables.
              """
        self.buckets = buckets
        self.PAD_ID = 0
        self.batch_size = batch_size
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.topk_n = topk_n
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # some parameters
        with tf.device(devices[0]):
            self.dropoutRate = tf.Variable(
                float(dropoutRate), trainable=False, dtype=dtype)
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

        #input layer
        # 将输入的每个batch的句子通过embedding 转换成向量。
        with tf.device(devices[0]):
            self.inputs = []
            self.inputs_embed = []
            self.input_embedding = tf.get_variable('input_embedding', [vocab_size, size], dtype = dtype)
            # 建立最大长度的inputs,每次处理一个输入单词，所以需要循环最长的句子的单词个数，
            # 把所有的输入单词转换成embedding再存入Input_embed中。
            for i in xrange(buckets[-1]):
                input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size], name = "input{}".format(i))
                input_embed = tf.nn.embedding_lookup(self.input_embedding, input_plhd)
                self.inputs.append(input_plhd)
                self.inputs_embed.append(input_embed)

        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)  #只需要告诉  h和c 的size就可以了
            # 建立输入的 dropout
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropoutRate)
            return cell

        # LSTM
        # with tf.device 可以让每个计算都绑定到不同的 gpu 上
        with tf.device(devices[1]):
            if num_layers == 1:
                single_cell = lstm_cell()
            else:
                single_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in xrange(num_layers)], state_is_tuple=True)
            single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.dropoutRate) #对输出做dropout
        # 建立单个 rnn，多层，带 dropout
        self.single_cell = single_cell

        # Output Layer
        with tf.device(devices[2]):
            self.targets = []
            self.target_weights = []

            self.output_embedding = tf.get_variable("output_embeddiing", [vocab_size, size], dtype=dtype)
            self.output_bias = tf.get_variable("output_bias", [vocab_size], dtype=dtype)
            #每次处理一个输出单词，所以需要循环最长的句子的单词个数，
            # 把所有的输出单词转换成embedding再存入targets和target_weights中。
            for i in xrange(buckets[-1]):
                self.targets.append(tf.placeholder(tf.int32,
                    shape=[self.batch_size], name = "target{}".format(i)))
                self.target_weights.append(tf.placeholder(dtype,
                    shape = [self.batch_size], name="target_weight{}".format(i)))

        # Model with buckets
        # 对于多 buckets 我们需要对于每个 buckets 都需要计算 loss 和 update 操作，生成 self.losses供下面使用。
        self.model_with_buckets(self.inputs_embed, self.targets, self.target_weights, self.buckets, single_cell, dtype, devices=devices)

        # train
        #根据self.losses，更新参数。
        with tf.device(devices[0]):
            params = tf.trainable_variables()
            # 不仅前向计算 forward, backward, update,计算backward和更新相关参数。
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                if withAdagrad:
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params, colocate_gradients_with_ops=True)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))
        #保存相关参数。
        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())

    #根据buckets建立不同的模型，并记录下self.losses.
    def model_with_buckets(self, inputs, targets, weights, buckets, cell, dtype, per_example_loss=False, name=None, devices=None):
        all_inputs = inputs + targets + weights
        losses = []  #每个bucket的loss分别加入到losses中。
        hts = []   #每个bucket的输出加入到hts中
        logits = []  #对每个bucket的输出hts计算wx+b 得到 logits，用于后续的softmax 和交叉熵。
        topk_values = []
        topk_indexes = []
        # initial state
        with tf.device(devices[1]):
            init_state = cell.zero_state(self.batch_size, dtype)

        # softmax
        with tf.device(devices[2]):
            softmax_loss_function = lambda x, y: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y)

        with tf.name_scope(name, "model_with_buckets", all_inputs):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                    # ht
                    with tf.device(devices[1]):
                        _hts, _ = tf.contrib.rnn.static_rnn(cell, inputs[:bucket], initial_state=init_state) # 通过  static_rnn 计算 输入  Ht
                        hts.append(_hts) #将每一次的输出 ht 加到 list中，每次的输出都是一个bucket的模型的输出。
                    #通过 hts 计算 Logits
                    with tf.device(devices[2]):
                        _logits = [tf.add(tf.matmul(ht, tf.transpose(self.output_embedding)), self.output_bias) for ht in _hts] #通过 hts 计算 Logits
                        logits.append(_logits)
                        if per_example_loss:  # 调用sequence_loss_by_example函数生成每一个bucket的loss。
                            losses.append(sequence_loss_by_example(
                                logits[-1], targets[:bucket], weights[:bucket],
                                softmax_loss_function=softmax_loss_function))
                        else:
                            losses.append(sequence_loss(
                                logits[-1], targets[:bucket], weights[:bucket],
                                softmax_loss_function=softmax_loss_function))
                        #每一个bucket预测的最后一个词可能会有topk_n个，都加入到topk_values中。
                        topk_value, topk_index = [], []
                        for _logits in logits[-1]:
                            value, index = tf.nn.top_k(tf.nn.softmax(_logits), self.topk_n, sorted=True)
                            topk_value.append(value)
                            topk_index.append(index)
                        topk_values.append(topk_value)
                        topk_indexes.append(topk_index)
        self.losses = losses
        self.hts = hts
        self.logits = logits
        self.topk_values = topk_values
        self.topk_indexes = topk_indexes

    def step(self,session, inputs, targets, target_weights, bucket_id, forward_only = False, dump_lstm = False):
        length = self.buckets[bucket_id]
        input_feed = {}
        for l in xrange(length):
            input_feed[self.inputs[l].name] = inputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # output_feed
        if forward_only:
            output_feed = [self.losses[bucket_id]]
            if dump_lstm:
                output_feed.append(self.states_to_dump[bucket_id])
        else:
            output_feed = [self.losses[bucket_id]]
            output_feed += [self.updates[bucket_id], self.gradient_norms[bucket_id]]
        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)
        if forward_only and dump_lstm:
            return outputs
        else:
            return outputs[0] # only return losses

    def get_batch(self, data_set, bucket_id, start_id=None):
        length = self.buckets[bucket_id]  # 选取的Bucket的句子长度，也就是需要补充到的长度。
        input_ids,output_ids, weights = [], [], []
        for i in xrange(self.batch_size):
            if start_id == None:
                word_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    word_seq = data_set[bucket_id][start_id + i]
                else:
                    word_seq = []
            word_input_seq = word_seq[:-1]  # without _EOS
            word_output_seq = word_seq[1:]  # without _GO

            target_weight = [1.0] * len(word_output_seq) + [0.0] * (length - len(word_output_seq))  #根据输出 配置 weight
            word_input_seq = word_input_seq + [self.PAD_ID] * (length - len(word_input_seq))
            word_output_seq = word_output_seq + [self.PAD_ID] * (length - len(word_output_seq))

            input_ids.append(word_input_seq)
            output_ids.append(word_output_seq)
            weights.append(target_weight)
        # Now we create batch-major vectors from the data selected above.
        #将上面的输入、输出以及weights 的list 变成矩阵的形式，也就是 Batch_size * bucekt的句子长度。
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output

        batch_input_ids = batch_major(input_ids)
        batch_output_ids = batch_major(output_ids)
        batch_weights = batch_major(weights)

        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True
        return batch_input_ids, batch_output_ids, batch_weights, finished

    def get_batch_test(self, data_set, bucket_id, start_id=None):
        length = self.buckets[bucket_id]
        word_inputs, positions, valids = [], [], []
        for i in xrange(self.batch_size):
            if start_id == None:
                word_seq = random.choice(data_set[bucket_id])
                valid = 1
                position = len(word_seq) - 1
            else:
                if start_id + i < len(data_set[bucket_id]):
                    word_seq = data_set[bucket_id][start_id + i]
                    valid = 1
                    position = len(word_seq) - 1
                else:
                    word_seq = []
                    valid = 0
                    position = length - 1

            word_input_seq = word_seq + [self.PAD_ID] * (length - len(word_seq))
            valids.append(valid)
            positions.append(position)
            word_inputs.append(word_input_seq)
        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output
        batch_word_inputs = batch_major(word_inputs)
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True
        return batch_word_inputs, positions, valids, finished

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".
  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:  #如果没有设置soft_max_loss 函数，就自己调用tf的相关函数计算 ht与 target的softmax 以及交叉熵。
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:                             #如果设置了相关函数，直接调用这个函数计算 ht与 target的softmax 以及交叉熵。
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)  #计算的loss与weight相乘，去掉加padding的loss

    log_perps = math_ops.add_n(log_perp_list)  #需要将一句话中每个词的loss都相加求和，得到最终的loss
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps

def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=False, softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
     Args:
       logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
       targets: List of 1D batch-sized int32 Tensors of the same length as logits.
       weights: List of 1D batch-sized float-Tensors of the same length as logits.
       average_across_timesteps: If set, divide the returned cost by the total
         label weight.
       average_across_batch: If set, divide the returned cost by the batch size.
       softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
         to be used instead of the standard softmax (the default if this is None).
       name: Optional name for this operation, defaults to "sequence_loss".
     Returns:
       A scalar float Tensor: The average log-perplexity per symbol (weighted).
     Raises:
       ValueError: If len(logits) is different from len(targets) or len(weights).
     """
    with tf.name_scope(name, "sequence_loss", logits + targets + weights):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            total_size = tf.reduce_sum(tf.sign(weights[0]))
            return cost / math_ops.cast(total_size, cost.dtype)
        else:
            return cost