运行于 Python  2.5
       tensorflow 1.2
1.main()
     （1）parsing_flags()  初始化参数
     （2）if FLAGS.mode == 'TRAIN':
                train()  转入  train()函数
     （3）if FLAGS.mode == 'FORCE_DECODE':
               batch_size = 1

2.train()
    （1）读入train数据和dev数据，获得 train_data_bucket,dev_data_bucket,_buckets
         train_data_bucket：[[s1,s1,s1,s2,s2],[s3,s3,s3,s3,s3,s4,s4,s4,s4,s5,s5],[s6,s6,s6]]
    （2）打印需要的信息
    （3） with tf.Session(config=config) as sess:
            1.创建模型 model = create_model(sess, run_options, run_metadata)
            2.显示变量 show_all_variables()
            3.调用 data_iterator.py 迭代生成数据 dite = DataIterator(model, train_data_bucket, len(train_buckets_scale), batch_size, train_buckets_scale)
            4.statistics during training  配置训练期间需要显示的信息。
            5.开始训练：
                当前训练步数 < 总步数时  或者  patience  > 0 时： 循环
                    1.数据输入到Mode中，得到损失量L,L = model.step(sess, inputs, outputs, weights, bucket_id)
                    2.if current_step % steps_per_report == 0：
                         显示一些运行时的信息。
                    3.if current_step % steps_per_checkpoint == 0：
                         达到半个epoch，计算ppx(dev)
                         如果 ppx(dev) < low_ppx  保存当前的best参数。patience = 初始值。
                         否则 patience -=1   learning_rate *= 0.5
                         如果 patience <10   则退出训练。

3.force_decode()
    预测，需要把 batch_size = 1

4.create_model(session, run_options, run_metadata)
    1.创建模型对象：model = SeqModel()
    2.ckpt = tf.train.get_checkpoint_state(FLAGS.saved_model_dir)
    3.恢复参数或者创建新的参数。




