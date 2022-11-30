
from munch import DefaultMunch

Config = {
    # 路径
    'data_path' : './my_data/download/W. South Pacific',
    #年份
    'year': "2022",

    'img_size' : 128, # 图片大小

    'batch_size_GAN' : 64, #GAN的batch size
    'epochs_GAN': 10000,   # GAN训练的轮数
    'lr_GAN': 0.0001,       # GAN的学习率（不要随便改）
    'latent_dim' : 32 ,    # 隐变量的维度
    'n_critic': 5,          # 训练D的次数 （不要随便改）
    'clip_value':0.01,      # 裁减D的参数（不要随便改）

    'load_G_name' : 'my_G3900.pth',  # 加载G的模型
    'epochs_lstm': 30000 ,             # LSTM训练的轮数
    'batch_size_lstm': 4 ,           # LSTM的batch （和序列有关，不要随便改）
    'lr_lstm': 0.01,                # LSTM的学习率
    'J': 8,                          # 用几张去预测一张
    'hidden_size' : 64,            # lstm menmory层的维度
    'layer': 2,                      # lstm 叠加的层数
}
dic_obj = DefaultMunch.fromDict(Config)