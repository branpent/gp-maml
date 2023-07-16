import torch
import torch.nn as nn
import warnings

import torchvision.transforms.functional
from torchmeta.datasets.helpers import (omniglot, miniimagenet, tieredimagenet, cifar_fs, fc100, cub, doublemnist,
                                        triplemnist)
from model import (MAML_ConvolutionalNeuralNetwork, ANIL_ConvolutionalNeuralNetwork, BOIL_ConvolutionalNeuralNetwork,
                   ReconstructionBoil_ConvolutionalNeuralNetwork, BasicBlock, ResNet)
from torchmeta.transforms import Categorical, ClassSplitter
from torchvision.transforms import Compose, Resize, ToTensor

def load_dataset(args, mode):
    seed = 1

    folder = '../data'
    ways = args.num_ways
    shots = args.num_shots
    test_shots = args.num_querys
    download = True
    shuffle = True

    if mode == 'meta_train':
        args.meta_train = True
        args.meta_val = False
        args.meta_test = False
    elif mode == 'meta_valid':
        args.meta_train = False
        args.meta_val = True
        args.meta_test = False
    elif mode == 'meta_test':
        args.meta_train = False
        args.meta_val = False
        args.meta_test = True

    if args.dataset == 'omniglot':
        dataset = omniglot(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download,
                           seed=seed)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download, seed=seed)
    elif args.dataset == 'cifar_fs':
        dataset = cifar_fs(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download, seed=seed)
    elif args.dataset == 'FC100':
        dataset = fc100(folder=folder,
                        shots=shots,
                        ways=ways,
                        shuffle=shuffle,
                        test_shots=test_shots,
                        meta_train=args.meta_train,
                        meta_val=args.meta_val,
                        meta_test=args.meta_test,
                        download=download, seed=seed)

    return dataset


def helper_with_default(klass, folder, shots, ways, shuffle=True,
                        test_shots=None, seed=None, defaults={}, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
                      'set in the helper function for the number of classes per task. '
                      'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        test_shots = shots

    dataset = klass(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
                            num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset


def load_model(args):
    algorithm = args.algorithm
    model = args.model

    if args.dataset == 'miniimagenet':
        wh_size = 5
    elif args.dataset == 'cifar_fs' or args.dataset == 'FC100':
        wh_size = 2
    elif args.dataset == 'omniglot':
        wh_size = 1

    if args.dataset in ('omniglot'):
        in_channels = 1
    else:
        in_channels = 3
    out_channels = args.num_ways
    hidden_size = args.hidden_size

    if algorithm + '_' + model == 'MAML_4conv':
        model = MAML_ConvolutionalNeuralNetwork(in_channels=in_channels, out_channels=out_channels,
                                                hidden_size=hidden_size, wh_size=wh_size)
    elif algorithm + '_' + model == 'ANIL_4conv':
        model = ANIL_ConvolutionalNeuralNetwork(in_channels=in_channels, out_channels=out_channels,
                                                hidden_size=hidden_size, wh_size=wh_size)
    elif algorithm + '_' + model == 'BOIL_4conv':
        model = BOIL_ConvolutionalNeuralNetwork(in_channels=in_channels, out_channels=out_channels,
                                                hidden_size=hidden_size, wh_size=wh_size)
    elif algorithm + '_' + model == 'RECONSTRUCTION_BOIL_4conv':
        model = ReconstructionBoil_ConvolutionalNeuralNetwork(in_channels=in_channels, out_channels=out_channels,
                                                              hidden_size=hidden_size, wh_size=wh_size)
    elif model == 'resnet':
        model = ResNet(algorithm=args.algorithm, keep_prob=1.0, avg_pool=True, drop_rate=0.0,
                       out_features=args.num_ways, wh_size=1)

    return model


def get_accuracy(logits, targets):
    _, prediction = torch.max(logits, dim=-1)
    return torch.mean(prediction.eq(targets).float())

def adaptivePicking(args, test_input, test_logits):
    soft_logits = torch.softmax(test_logits, dim=1)
    soft_logits_max = torch.max(soft_logits, dim=1)
    fake_label = torch.argmax(soft_logits, dim=1)

    fake_indices = torch.topk(soft_logits_max.values, k=args.pic_num, dim=0).indices
    fakes = [[]] * args.num_ways
    for i in range(args.num_ways):
        fakes[i] = [idx for idx in fake_indices if fake_label[idx] == i]

    fake_input, fake_target = None, None

    min_len = 99999
    for i in range(args.num_ways):
        if len(fakes[i]) < min_len:
            min_len = len(fakes[i])

    for i in range(min_len):
        for idx in range(args.num_ways):
            if fake_input == None:
                fake_input = test_input[fakes[idx][i]].unsqueeze(0)
            else:
                fake_input = torch.cat([fake_input, test_input[fakes[idx][i]].unsqueeze(0)])

    not_selected_input, not_selected_target = None, None

    for i in range(args.num_ways):
        if len(fakes[i]) > min_len:
            for j in range(min_len, len(fakes[i])):
                if not_selected_input == None:
                    not_selected_input = test_input[fakes[i][j]].unsqueeze(0)
                    not_selected_target = fake_label[fakes[i][j]].unsqueeze(0)
                else:
                    not_selected_input = torch.cat([not_selected_input, test_input[fakes[i][j]].unsqueeze(0)])
                    not_selected_target = torch.cat([not_selected_target, fake_label[fakes[i][j]].unsqueeze(0)])

    fake_target = torch.tensor([i for i in range(args.num_ways)] * min_len , dtype=torch.long, device=args.device)

    return fake_input, fake_target, not_selected_input, not_selected_target

def _get_recon_dist(query, support):
    reg = support.size(1) / support.size(2)

    st = support.permute(0, 2, 1)  # way, d, hw
    sst = support.matmul(st)
    m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0)).inverse()
    hat = st.matmul(m_inv).matmul(support)

    Q_bar = query.matmul(hat)
    dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
    return dist

def get_entropy(logits):
    return -1.0 * torch.mean(torch.sum(torch.softmax(logits, 1) * torch.log_softmax(logits, 1), 1))
