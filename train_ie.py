from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


import torch.nn as nn
import fairseq
from d2l import torch as d2l

import os
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Script with YAML config.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file (YAML)')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_configs(cli_args, yaml_config):
    # Assuming cli_args is the Namespace object returned by argparse
    # And yaml_config is a dictionary loaded from the YAML file
    config = vars(cli_args)  # Convert Namespace to dict
    config.update(yaml_config)  # YAML config overrides CLI args
    return config


# Parse command line arguments
cli_args = parse_args()

# Load configuration from YAML file
yaml_config = load_config(cli_args.config)

# Optionally, merge CLI args with YAML configs
config = merge_configs(cli_args, yaml_config)

# Now use your config
print(config)


##### DATA iter Prepareing

#### parameters
test_session = config['test_session']
print("Processing test_session: ", test_session)
batch_size = config['batch_size']
label_list = config['label_list']
fusing_folder = config['fusing_folder']
num_lables = len(label_list)
####
front_path = config['front_path']
data_files = {
    "train": "/home/felix/Aopp/FINAL/data/IEMOCAP/{}/test{}/train.csv".format(fusing_folder, test_session),
    "test": "/home/felix/Aopp/FINAL/data/IEMOCAP/{}/test{}/test.csv".format(fusing_folder, test_session),
}

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
   
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target
    
def speech_file_to_array_fn(path):
    ## add path hear

    speech_array, sampling_rate = torchaudio.load(os.path.join(front_path, path))
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array).squeeze().numpy()
    return torch.tensor(speech)

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label

def load_dataset_csv(path, label_list):
    temp_df = pd.read_csv(path, sep="\t", encoding="utf-8")


    speech_list = [speech_file_to_array_fn(file_path) for file_path in temp_df['path']]
    target_list = [label_to_id(label, label_list) for label in temp_df['emotion']]

    return MyDataset(speech_list,target_list)

def get_dataloader_workers():
    return 16

def collate_fn(batch):
    # Assume that each element in "batch" is a tuple (data, label)
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences in this batch
    data = pad_sequence(data, batch_first=True)

    return data, torch.tensor(labels)


train_iter = data.DataLoader(load_dataset_csv(data_files['train'], label_list), batch_size, shuffle=True,
                             num_workers=get_dataloader_workers(), collate_fn=collate_fn)

test_iter = data.DataLoader(load_dataset_csv(data_files['test'], label_list), batch_size, shuffle=True,
                             num_workers=get_dataloader_workers(), collate_fn=collate_fn)

##### DATA iter Prepareing End

##### training

#### parameters
o_model_path = config['o_model_path']

load_path = config['load_path']
load_path = load_path + test_session
lr1 = config['lr1']
lr2 = config['lr2']
epoch = config['epoch']

####

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec downstream task."""

    def __init__(self, dense_size, dropout_rate, num_labels):
        super().__init__()
        self.dense = nn.Linear(dense_size, dense_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(dense_size, num_labels)

    def forward(self, features):
        features = torch.mean(features, dim=-1)
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def train_net(net, optimizer, train_iter, test_iter, start_epoch, num_epochs, device, losses, train_accs, test_accs, best_test_acc):
    net.to(device)
    loss = nn.CrossEntropyLoss()
    
    best = best_test_acc
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
            if i == num_batches - 1:
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                losses.append(metric[0] / metric[2])
                train_accs.append(metric[1] / metric[2])
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        test_accs.append(test_acc)

        if test_acc > best:
            best = test_acc
            torch.save({
                    'model': net,
                    'optimizer': optimizer,
                    'epoch': epoch,
                    'losses': losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs,
                    'best_test_acc': best
                }, os.path.join(load_path, 'best.pth'))
            
        if epoch % 20 == 0:
            torch.save({
                    'model': net,
                    'optimizer': optimizer,
                    'epoch': epoch,
                    'losses': losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs,
                    'best_test_acc': best
                }, os.path.join(load_path, 'last.pth'))


        print("EPOCH {}: loss {:.3f}, train acc {:.3f}, test acc {:.3f}".format(epoch, train_l, train_acc, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

### train start



if not os.path.exists(load_path):
    os.makedirs(load_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([o_model_path])
    model = model[0]
    
    dense_size = model.feature_extractor.conv_layers[-1][0].out_channels
    net = nn.Sequential(model.feature_extractor, Wav2Vec2ClassificationHead(dense_size, 0.05, num_lables))
    net[1].apply(init_weights)

    if lr1 == 0:
        for param in net[0].parameters():
            param.requires_grad = False
    
    params_to_optimize = [
        {"params": net[0].parameters(), "lr": lr1},
        {"params": net[1].parameters(), "lr": lr2},
    ]

    optimizer = torch.optim.Adam(params_to_optimize)
    start_epoch = 0

    losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0
else:
    checkpoint = torch.load(os.path.join(load_path, 'last.pth'))

    net = checkpoint['model']
    optimizer = checkpoint['optimizer']
    start_epoch = checkpoint.get('epoch', 1)

    losses = checkpoint.get('losses', 1)
    train_accs = checkpoint.get('train_accs', 1)
    test_accs = checkpoint.get('test_accs', 1)
    best_test_acc = checkpoint.get('best_test_acc', 0)



train_net(net, optimizer, train_iter, test_iter, start_epoch + 1, epoch, d2l.try_gpu(), losses, train_accs, test_accs, best_test_acc)
