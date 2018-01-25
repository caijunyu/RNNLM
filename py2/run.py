# -*- coding: utf-8 -*-
import os
import sys
import time
import logging
import math

import numpy as np
import tensorflow as tf
from seqModel import SeqModel
from data_iterator import DataIterator
from tensorflow.python.client import timeline
from data_util import read_train_dev, get_real_vocab_size, read_test, get_vocab_path

# mode
tf.app.flags.DEFINE_string("mode", "TRAIN", "TRAIN|FORCE_DECODE|BEAM_DECODE|DUMP_LSTM")

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("model_dir", "./model", "model_dir/data_cache/n model_dir/saved_model; model_dir/log.txt .")
# tf.app.flags.DEFINE_string("train_path", "/home/cai888/RNNLMCAI/data/ptb/train", "the absolute path of raw train file.")
tf.app.flags.DEFINE_string("train_path", "./train", "the absolute path of raw train file.")
tf.app.flags.DEFINE_string("dev_path", "./valid", "the absolute path of raw dev file.")
tf.app.flags.DEFINE_string("test_path", "./test", "the absolute path of raw test file.")
tf.app.flags.DEFINE_string("force_decode_output", "force_decode.txt",
                           "the file name of the score file as the output of force_decode. The file will be put at model_dir/force_decode_output")

# tuning hypers
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.8,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training/evaluation.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,"Maximum number of epochs in training.")
tf.app.flags.DEFINE_integer("L", 30, "max length")
tf.app.flags.DEFINE_integer("n_bucket", 10,"num of buckets to run.")
tf.app.flags.DEFINE_integer("patience", 10, "exit if the model can't improve for $patence evals")

# devices
tf.app.flags.DEFINE_string("N", "000", "GPU layer distribution: [input_embedding, lstm, output_embedding]")

# training parameter
tf.app.flags.DEFINE_boolean("withAdagrad", True,"withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,"withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,"save Model at each checkpoint.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

# GPU configuration
tf.app.flags.DEFINE_boolean("allow_growth", False, "allow growth")

# for beam_decode
tf.app.flags.DEFINE_integer("topk", 3, "topk")

FLAGS = tf.app.flags.FLAGS

def create_model(session, run_options, run_metadata):
    devices = get_device_address(FLAGS.N)
    dtype = tf.float32
    model = SeqModel(FLAGS._buckets,
                     FLAGS.size,
                     FLAGS.real_vocab_size,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     withAdagrad=FLAGS.withAdagrad,
                     dropoutRate=FLAGS.keep_prob,
                     dtype=dtype,
                     devices=devices,
                     topk_n=FLAGS.topk,
                     run_options=run_options,
                     run_metadata=run_metadata
                     )
    ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    if FLAGS.mode == "DUMP_LSTM" or FLAGS.mode == "BEAM_DECODE" or FLAGS.mode == 'FORCE_DECODE' or (not FLAGS.fromScratch) and ckpt:
        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def get_device_address(s):
    add = []
    if s == "":
        for i in range(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]

    return add

def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)

def train():
    #1.读入train数据和dev数据
    mylog_section('READ DATA')
    train_data_bucket,dev_data_bucket,_buckets,vocab_path = read_train_dev(FLAGS.data_cache_dir, FLAGS.train_path,FLAGS.dev_path, FLAGS.vocab_size, FLAGS.L,FLAGS.n_bucket)
    ##########以下是打印需要的信息 start #####################
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    # 计算总共要处理的tokens个数
    train_n_tokens = np.sum([np.sum([len(sentence) for sentence in bucket]) for bucket in train_data_bucket])

    # train_data_bucket
    train_bucket_sizes = [len(train_data_bucket[index]) for index in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    # 计算累计值,用于计算bucket，在 data_iterator中随机生成一个0-1的数，这里的train_buckets_scale根据每个bucket中句子数量的不同，切分成不同的权重[0.1,0.3,0.5,0.8,1]
    # 当随机的0-1的数落到上述权重的某个区间，那么就选哪个bucket。
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    dev_bucket_sizes = [len(dev_data_bucket[index]) for index in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))

    mylog_section("REPORT")
    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_checkpoint = int(steps_per_epoch / 2)  #每半个epoch　验证一次模型
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets: {}".format(FLAGS._buckets))
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))
    ##########打印需要的信息　end #####################

    mylog_section("IN TENSORFLOW")

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = FLAGS.allow_growth
    with tf.Session(config=config) as sess:
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog_section("MODEL/SUMMARY/WRITER")
        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, run_options, run_metadata)

        mylog_section("All Variables")
        show_all_variables()

        # Data Iterators
        mylog_section("Data Iterators")
        dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
        iteType = 0
        if iteType == 0:
            mylog("Itetype: withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("Itetype: withSequence")
            ite = dite.next_sequence()

        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        low_ppx = float("inf")
        steps_per_report = 30
        n_targets_report = 0
        report_time = 0
        n_valid_sents = 0
        n_valid_words = 0
        patience = FLAGS.patience

        mylog_section("TRAIN")
        while current_step < total_steps:
            # start
            start_time = time.time()
            # data and train
            inputs, outputs, weights, bucket_id = ite.next()   #训练数据

            L = model.step(sess, inputs, outputs, weights, bucket_id)

            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += L
            current_step += 1
            # 此处 weights 等数据的格式是 len(weights) == 句子长度
            # len(weights[0]) 是 batch size
            n_valid_sents += np.sum(np.sign(weights[0]))
            n_valid_words += np.sum(weights)
            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(weights)

            #显示信息
            if current_step % steps_per_report == 0:
                sect_name = "STEP {}".format(current_step)
                msg = "StepTime: {:.2f} sec Speed: {:.2f} targets/s Total_targets: {}".format(
                    report_time / steps_per_report, n_targets_report * 1.0 / report_time, train_n_tokens)
                mylog_line(sect_name, msg)

                report_time = 0
                n_targets_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()

            #达到半个epoch，计算ppx(dev)
            if current_step % steps_per_checkpoint == 0:
                i_checkpoint = int(current_step / steps_per_checkpoint)
                # train_ppx
                loss = loss / n_valid_words
                train_ppx = math.exp(float(loss)) if loss < 300 else float("inf")
                learning_rate = model.learning_rate.eval()

                # dev_ppx
                dev_loss, dev_ppx = evaluate(sess, model, dev_data_bucket)

                # report
                sect_name = "CHECKPOINT {} STEP {}".format(i_checkpoint, current_step)
                msg = "Learning_rate: {:.4f} Dev_ppx: {:.2f} Train_ppx: {:.2f}".format(learning_rate, dev_ppx,train_ppx)
                mylog_line(sect_name, msg)

                # save model per checkpoint
                if FLAGS.saveCheckpoint:
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "model")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=i_checkpoint, write_meta_graph=False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time() - s, checkpoint_path)
                    mylog_line(sect_name, msg)

                # save best model
                if dev_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = dev_ppx
                    checkpoint_path = os.path.join(FLAGS.saved_model_dir, "best")
                    s = time.time()
                    model.best_saver.save(sess, checkpoint_path, global_step=0, write_meta_graph=False)
                    msg = "Model saved using {:.2f} sec at {}".format(time.time() - s, checkpoint_path)
                    mylog_line(sect_name, msg)
                else:
                    patience -= 1
                    #每次当 dev_ppx >= low_ppx时 学习步长减半
                    sess.run(model.learning_rate_decay_op)
                    msg = 'dev_ppx:{}, low_ppx:{}'.format(str(dev_ppx), str(low_ppx))
                    mylog_line(sect_name, msg)
                    msg = 'dev_ppx >= low_ppx，patience ={}, learning_reate ={}'.format(str(patience), str(model.learning_rate.eval()))
                    mylog_line(sect_name, msg)

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents, n_valid_words = 0.0, 0.0, 0, 0


#达到半个epoch，计算ppx(dev)
def evaluate(sess, model, data_set):
    # Run evals on development set and print their perplexity/loss.
    sess.run(model.dropout10_op)# 验证的时候dropout设置为1，也就是不dropout
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size

    dite = DataIterator(model, data_set, len(FLAGS._buckets), batch_size, None)
    ite = dite.next_sequence(stop=True)

    for inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, inputs, outputs, weights, bucket_id, forward_only=True)
        loss += L
        n_steps += 1
        n_valids += np.sum(weights)
    loss = loss / (n_valids)
    ppx = math.exp(loss) if loss < 300 else float("inf")
    sess.run(model.dropoutAssign_op)  #验证结束需要将 dropout恢复原来的设置。
    return loss, ppx

#预测
def force_decode():
    # force_decode it: generate a file which contains every score and the final score;
    mylog_section("READ DATA")
    #读入test数据,test不需要新建立词典，直接调用建立好的词典就可以了。
    test_data_bucket, _buckets, test_data_order = read_test(FLAGS.data_cache_dir, FLAGS.test_path,get_vocab_path(FLAGS.data_cache_dir), FLAGS.L,FLAGS.n_bucket)
    vocab_path = get_vocab_path(FLAGS.data_cache_dir)
    real_vocab_size = get_real_vocab_size(vocab_path)

    FLAGS._buckets = _buckets
    FLAGS.real_vocab_size = real_vocab_size

    test_bucket_sizes = [len(test_data_bucket[b]) for b in range(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog_section("REPORT")
    mylog("real_vocab_size: {}".format(FLAGS.real_vocab_size))
    mylog("_buckets:{}".format(FLAGS._buckets))
    mylog("FORCE_DECODE:")
    mylog("total: {}".format(test_total_size))
    mylog("bucket_sizes: {}".format(test_bucket_sizes))

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = FLAGS.allow_growth

    mylog_section("IN TENSORFLOW")
    with tf.Session(config=config) as sess:
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, run_options, run_metadata)

        mylog_section("All Variables")
        show_all_variables()

        sess.run(model.dropoutRate.assign(1.0))
        batch_size = FLAGS.batch_size
        mylog_section("Data Iterators")
        dite = DataIterator(model, test_data_bucket, len(_buckets), batch_size, None, data_order=test_data_order)
        ite = dite.next_original()

        fdump = open(FLAGS.score_file, 'w')
        i_sent = 0

        mylog_section("FORCE_DECODING")
        for inputs, outputs, weights, bucket_id in ite:
            # inputs: [[_GO],[1],[2],[3],[_EOS],[pad_id],[pad_id]]
            # positions: [4]
            mylog("--- decoding {}/{} sent ---".format(i_sent, test_total_size))
            i_sent += 1
            L = model.step(sess, inputs, outputs, weights, bucket_id, forward_only=True, dump_lstm=False)
            mylog("LOSS: {}".format(L))
            fdump.write("{}\n".format(L))
        fdump.close()

def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)

def mylog_section(section_name):
    mylog("======== {} ========".format(section_name))

def log_flags():
    members = FLAGS.__dict__['__flags'].keys()
    mylog_section("FLAGS")
    for attr in members:
        mylog("{}={}".format(attr, getattr(FLAGS, attr)))

def mylog_line(section_name, message):
    mylog("[{}] {}".format(section_name, message))

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def parsing_flags():
    FLAGS.data_cache_dir = os.path.join(FLAGS.model_dir,'data_cache')
    FLAGS.saved_model_dir = os.path.join(FLAGS.model_dir, "saved_model")
    FLAGS.summary_dir = FLAGS.saved_model_dir

    mkdir(FLAGS.model_dir)
    mkdir(FLAGS.data_cache_dir)
    mkdir(FLAGS.saved_model_dir)
    mkdir(FLAGS.summary_dir)

    # for logs
    log_path = os.path.join(FLAGS.model_dir, "log.{}.txt".format(FLAGS.mode))  #log地址
    filemode = 'w' if FLAGS.fromScratch else "a"
    logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode=filemode)

    log_flags()

def main():
    #读取超参数
    parsing_flags()

    if FLAGS.mode == 'TRAIN':
        train()

    if FLAGS.mode == 'FORCE_DECODE':
        mylog(
            "\nWARNING: \n 1. The output file and original file may not align one to one, because we remove the lines whose lenght exceeds the maximum length set by -L \n 2. The score is -sum(log(p)) with base e and includes EOS. \n")

        FLAGS.batch_size = 1
        FLAGS.score_file = os.path.join(FLAGS.model_dir, FLAGS.force_decode_output)  #预测结果的输出文件地址
        # FLAGS.n_bucket = 1
        force_decode()



if __name__ == '__main__':
    main()