import torch
import argparse
# import sys
# print(sys.path.append('./'))
import sys, os
base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)
sys.path.append(base_path)
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import model as models
import pickle
from config import cfg
from torchnet import meter
from data.dataset import ValData,TrainData



parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--model_save', help='Pretrained model_save or nothing',
                    type=str, default=None)
parser.add_argument('--mode', help = 'train or test',type = str,
                    default= 'test')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print("Producing data...")

def numpy_to_tensor(x):
    return torch.unsqueeze(torch.from_numpy(x), dim=1)

def train(**kwargs):
    cfg._parse(kwargs)
    model = getattr(models, cfg.model)
    if cfg.load_model_param_path:
        model.load_state_dict(torch.load(cfg.load_model_param_path))
    model.to(cfg.device)

    train_data = TrainData(cfg.train_data_root)

    lr = cfg.lr
    # optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr,betas=(0.9, 0.99),weight_decay= cfg.weight_decay)
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e1
    for epoch in range(1, cfg.max_epoch+1):
        loss_meter.reset()
        for j in range(1):
            X_train, Y_train = numpy_to_tensor(train_data[0][j]), numpy_to_tensor(train_data[1][j])
            if cfg.use_gpu:
                X_train,Y_train = X_train.cuda(), Y_train.cuda()

            # print(X_train.shape)

            batch_idx = 1
            total_loss = 0

            for i in range(3600):

                x, y = X_train[i:(i + cfg.batch_size)], Y_train[i]
                optimizer.zero_grad()
                x, y = x.float(), y.float()
                output = model(x)
                # print(output.shape)
                output = output.permute(1,0)
                # print('output:', output.shape)
                loss = F.mse_loss(output, y)
                # print('loss', loss)
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                batch_idx += 1
                loss_meter.add(loss.item())
                total_loss +=loss.item()

                if batch_idx % args.log_interval == 0:

                    cur_loss = total_loss / args.log_interval
                    processed = min(i+cfg.batch_size, X_train.size(0))
                    print('Train Epoch: {:2d}\tnum:{:4d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.6f}\tLoss: {:.6f}'.format(
                        epoch,j,processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
                    total_loss = 0
        # torch.save(model, './output/model_save/model_{}.pth'.format(epoch))
        torch.save(model.state_dict(),'./output/models_param/model_param_{}.pth'.format(epoch))
        if loss_meter.value()[0] >= previous_loss:
            lr = lr * cfg.lr_decay
        if epoch % 6 ==0:
            lr = lr * cfg.lr_decay
        if epoch % 2 == 0:
            val(model)

        previous_loss = loss_meter.value()[0]


@torch.no_grad()
def val(model):
    print('-----------begin val-------------')
    model.eval()
    output = torch.Tensor(3600, 1, 30)
    X_val, Y_val = ValData(cfg.test_data_root)
    X_val, Y_val = numpy_to_tensor(X_val[0]), numpy_to_tensor(Y_val[0])


    if cfg.use_gpu:
        X_val, Y_val ,output= X_val.cuda(), Y_val.cuda(), output.cuda()
    for i in range(3600):
        # print(X_val[i: i+cfg.batch_size].shape)
        output[i] = model(X_val[i: i+cfg.batch_size].float()).permute(1, 0)
    test_loss = F.mse_loss(output.float(), Y_val.float())
    print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
    model.train()
    return test_loss.item()

@torch.no_grad()
def test(**kwargs):
    cfg._parse(kwargs)

    model = getattr(models, cfg.model).eval()
    if cfg.load_model_param_path:
        model.load_state_dict(torch.load(cfg.load_model_param_path))
    model.to(cfg.device)

    # test_data = torch.from_numpy(np.array(ValData(cfg.test_data_root)[0]))
    output = torch.Tensor(3600, 1, 30)
    test_data = numpy_to_tensor(ValData(cfg.test_data_root)[0][0])
    # print('test_data',test_data)
    if cfg.use_gpu:
        test_data, output = test_data.cuda(), output.cuda()
    for i in range(3600):

        output[i] = model(test_data[i: i + 30].float()).permute(1, 0)
        # print(output[i].shape)

    output = torch.squeeze(output, dim=1)
    output = output.cpu().numpy()
    write_fm_y(output, cfg.result_file)

def write_fm_y(output,file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(output, fp)
        fp.close()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        test()




