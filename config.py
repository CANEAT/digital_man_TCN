import warnings
import torch as t


class DefaultConfig(object):
    env = 'default'
    model = 'TCN'

    input_channels = 1  # 输入深度
    n_classes = 30  # 输出维度
    batch_size = 30
    nhid = 30
    levels = 8
    channel_sizes = [nhid] * levels#深度
    kernel_size = 7

    # train_data_root ='G:\\研究生课程\\deecamp2019\\data\\deecamp_51\\DeeCamp_52\\DeeCamp\\values_0701all\\values'
    # test_data_root ='G:\研究生课程\deecamp2019\data\deecamp_51\DeeCamp_52\DeeCamp\\values_0701all\\test'
    train_data_root = 'E:\\项目\\deecamp\\data\\DeeCamp_52\\DeeCamp\\values_0701all\\values'#Lab COMPUTER
    test_data_root = 'E:\\项目\\deecamp\\data\\DeeCamp_52\\DeeCamp\\TestData'#Lab COMPUTER
    # test_data_root = '/media/zsdx/D/jameszhu/DATA_MAN/DeeCamp/TestData'#Lab Servers
    # train_data_root = '/media/zsdx/D/jameszhu/DATA_MAN/DeeCamp/values_0701all/values'#Lab Servers
    load_model_param_path = 'E:\\项目\\deecamp\\code\\Digital_man_TCN_pytorch1\\Digital_man\output\\models_param\\model_param_2.pth'
    # load_model_param_path =None
    use_gpu = True

    result_file = './output/fmy_result/result_2.fm_y'

    dropout = 0

    max_epoch = 16
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5
    # cfg.device = t.device('cuda') if cfg.use_gpu else t.device('cpu')

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        cfg.device = t.device('cuda') if cfg.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

cfg = DefaultConfig()