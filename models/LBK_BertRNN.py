# _*_ coding: UTF-8 _*_
# Author LBK
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'LBK_BertRNN'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 校验集
        self.dev_path = dataset + '/data/dev.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # dataset
        self.datasetpkl = dataset + '/data/dataset.pkl'

        # 类别名单
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt'
        ).readlines()]
        # 模型训练保存类路径
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000 batch 效果还没有提升,提前结束训练
        self.require_improvement = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch 数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # 序列长度 padding_size  每句话处理的长度  短填长截
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练模型位置
        self.bert_path = './bert_pretrain'
        # Bert 的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # Bert 的 隐藏层数量
        self.hidden_size = 768

        # RNN 的隐藏层数量
        self.rnn_hidden = 256
        # RNN 数量
        self.rnn_layers = 2
        # dropout
        self.dropout = 0.5


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # input_size, hidden_size, num_layers, batch_first, dropout, bidirectional
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.rnn_layers,
                            batch_first=True, dropout=config.dropout, bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.rnn_hidden*2, config.num_classes)

    def forward(self, x):
        """x: [ids, seq_len, mask]"""
        context = x[0]  # 对应输入句子 shape[128, 32]
        mask = x[2]  # 对padding部分进行mask shape[128, 32]
        # encoder_out的shape[128, 32, 768]
        # text_cls的shape[128, 768]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out的shape[128, 32, 512]  c_n == h_n == [4, 128, 256]
        out, _ = self.lstm(encoder_out)

        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

