
head="""
ROOT_DIR=/home/cai888/RNNLMCAI/
PY=$ROOT_DIR/py2/run.py
MODEL_DIR=$ROOT_DIR/model/__id__
DATA_DIR=$ROOT_DIR/data/__data_dir__/
TRAIN_PATH=$DATA_DIR/train
DEV_PATH=$DATA_DIR/valid
TEST_PATH=$DATA_DIR/test

__cmd__
"""
train_cmd = "python $PY --mode TRAIN --train_path $TRAIN_PATH --dev_path $DEV_PATH --model_dir $MODEL_DIR "

def main(acct=0):
    def name(val):
        return val, ""

    def model_dir(val):
        return "", "--model_dir {}".format(val)

    def batch_size(val):
        return "", "--batch_size {}".format(val)

    def size(val):
        return "h{}".format(val), "--size {}".format(val)

    def dropout(val):
        return "d{}".format(val), "--keep_prob {}".format(val)

    def learning_rate(val):
        return "l{}".format(val), "--learning_rate {}".format(val)

    def n_epoch(val):
        return "", "--n_epoch {}".format(val)

    def num_layers(val):
        return "n{}".format(val), "--num_layers {}".format(val)

    def L(val):
        return "", "--L {}".format(val)

    def vocab_size(val):
        return "", "--vocab_size {}".format(val)

    def n_bucket(val):
        return '', "--n_bucket {}".format(val)

    funcs = [name, n_bucket, batch_size,  # 0
             size, dropout, learning_rate,  # 3
             n_epoch, num_layers, L,  # 6
             vocab_size]  # 9

    template = ["ptb", 10, 64,  # 0
                300, 0.5, 1.0,  # 3
                40, 2, 110,  # 6
                10050]  # 9

    params = []

    #for ptb
    _sizes = [300,600,329]
    _num_layers = [[1,2,3],[1,2,3],[1]]
    _dropouts = [0.5,1.0]
    _learning_rates = [0.5,1.0]
    for i, _size in enumerate(_sizes):
        ns = _num_layers[i]
        for _n in ns:
            for _d in _dropouts:
                for _l in _learning_rates:
                    temp = list(template)
                    temp[0] = 'ptb'
                    temp[3] = _size
                    temp[7] = _n
                    temp[4] = _d
                    temp[5] = _l
                    params.append(temp)

    def get_name_cmd(paras):
        name = ""
        cmd = []
        for func, para in zip(funcs, paras):
            n, c = func(para)
            name += n
            cmd.append(c)

        name = name.replace(".", '')

        cmd = " ".join(cmd)
        return name, cmd

    # train
    for para in params:
        name, cmd = get_name_cmd(para)
        cmd = train_cmd + cmd

        # for train
        fn = "../jobs/{}.sh".format(name)
        f = open(fn, 'w')
        content = head.replace("__cmd__", cmd)
        content = content.replace("__id__", name)
        content = content.replace("__data_dir__", para[0])
        f.write(content)
        f.close()

if __name__ == "__main__":
    main()