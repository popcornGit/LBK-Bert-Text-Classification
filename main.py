# _*_ coding: UTF-8 _*_
# Author LBK
import torch
import time
import numpy as np
from importlib import import_module
import argparse
import utils
import train


parser = argparse.ArgumentParser(description='LBK_Bert_Text_classification')
parser.add_argument('--model', type=str, default='LBK_ERNIEDPCNN', help='choose a model LBK_Bert LBK_BertCNN, LBK_BertRNN, LBK_BertRCNN, LBK_BertDPCNN, LBK_ERNIE, LBK_ERNIEDPCNN')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集地址
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    # 保证每次运行结果一致
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.build_dataset(config)

    train_iter = utils.build_iterator(train_data, config)
    dev_iter = utils.build_iterator(dev_data, config)
    test_iter = utils.build_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始前,准备数据时间:", time_dif)

    # 模型训练, 评估与测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    # train.test(config, model, test_iter)
