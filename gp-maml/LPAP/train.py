import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
import random
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d)
from torchmeta.utils.gradient_based import gradient_update_parameters
import model as m
import itertools
from utils import get_accuracy, load_dataset, load_model, adaptivePicking

class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

def meta_conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class BOIL_Encoder(MetaModule):
    def __init__(self, in_channels, out_channels, hidden_size=64, wh_size = 1):

        super(BOIL_Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_channels
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            meta_conv3x3(in_channels, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size),
            meta_conv3x3(hidden_size, hidden_size)
        )

    def forward(self, inputs, params=None):

        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        return features

class LabelPropagation(nn.Module):
    def __init__(self, args = None):
        super(LabelPropagation, self).__init__()

        self.device = args.device
        self.args = args
        if args.dataset == 'omniglot':
            self.encoder = BOIL_Encoder(1, args.num_ways, args.hidden_size)
        else:
            self.encoder = BOIL_Encoder(3, args.num_ways, args.hidden_size)
        self.relation = m.RelationNetWork(args)
        self.alpha = torch.tensor(0.99, requires_grad= False, device=args.device)

    def forward(self, inputs, params = None):
        eps = np.finfo(float).eps

        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        # Step1: Embedding
        inp = torch.cat((support, query), 0)
        if params!=None:
            if self.args.dataset == 'miniimagenet':
                emb_all = self.encoder(inp, params=params).view(-1, 1600)
            else:
                emb_all = self.encoder(inp, params = params).view(-1, 256)
        else:
            emb_all = self.encoder(inp).view(-1, 256)
        N, d = emb_all.shape[0], emb_all.shape[1]

        # Step2: Graph Construction
        self.sigma   = self.relation(emb_all)
        emb_all = emb_all / (self.sigma + eps)  # N*d
        emb1 = torch.unsqueeze(emb_all, 1)  # N*1*d
        emb2 = torch.unsqueeze(emb_all, 0)  # 1*N*d
        W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
        W = torch.exp(-W / 2)

        topk, indices = torch.topk(W, 20)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, kNN graph
        # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
        W = W * mask

        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(num_classes * num_queries, num_classes).to(self.device)
        # yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).to(self.device)
        y = torch.cat((ys, yu), 0)
        F = torch.matmul(torch.inverse(torch.eye(N).to(self.device) - self.alpha * S + eps), y)
        Fq = F[num_classes * num_support:, :]  # query predictions

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().to(self.device)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()
        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc, Fq

def main(mode, iteration=None):
    dataset = load_dataset(args, mode)

    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model.to(device = args.device)
    model.train()
    tpn.to(device=args.device)
    tpn.train()

    # learning rate
    if iteration < 100:
        meta_lr = 1e-3
    elif iteration >= 100 and iteration < 200:
        meta_lr = 1e-4
    elif iteration >= 200:
        meta_lr = 1e-5

    meta_optimizer = torch.optim.Adam(itertools.chain(model.parameters(), tpn.parameters()), lr = meta_lr)

    if mode == 'meta_train':
        total = args.train_batches
    elif mode == 'meta_valid':
        total = args.valid_batches
    elif mode == 'meta_test':
        total = args.test_batches

    loss_logs, accuracy_logs = [], []

    fake_pic_num = []
    tpn_acc_log = []

    with tqdm(dataloader, total=total) as pbar:
        for batch_idx, batch in enumerate(pbar):

            if batch_idx >= total:
                break

            tpn.zero_grad()
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)

            tpn_acc = torch.tensor(0., device= args.device)
            pic_num = torch.tensor(0., device=args.device)

            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):

                _, train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)
                model.zero_grad()
                params = gradient_update_parameters(model, inner_loss,step_size=0.5)

                # change to one-hot
                train_onehot_label = torch.zeros(train_target.shape[0], args.num_ways, device=args.device).scatter(1,
                                                                                                    train_target.view(
                                                                                                        -1, 1),
                                                                                                    1.)
                test_onehot_label = torch.zeros(test_input.shape[0], args.num_ways, device = args.device).scatter(1,
                                                                                                    test_target.view(
                                                                                                        -1, 1),
                                                                                                    1.)

                inputs = [train_input, train_onehot_label, test_input, test_onehot_label]

                tpn_loss, t_acc, tpn_test_logit = tpn(inputs, params = params)
                tpn_acc += t_acc

                if mode == 'meta_valid' or mode =='meta_test' or mode =='meta_train':
                    fake_input, fake_target, _, _ = adaptivePicking(args, test_input, tpn_test_logit)
                    if fake_input!=None:
                        pic_num += fake_input.shape[0]
                        pro_train_input = torch.cat([train_input, fake_input])
                        pro_train_target = torch.cat([train_target, fake_target])
                        _, pro_train_logit = model(pro_train_input)
                        pro_inner_loss = F.cross_entropy(pro_train_logit, pro_train_target)
                        model.zero_grad()
                        pro_params = gradient_update_parameters(model, pro_inner_loss, step_size=0.5)
                        _, test_logit = model(test_input, params = pro_params)
                    else:
                        _, test_logit = model(test_input, params = params)
                else:
                    _, test_logit = model(test_input, params = params)


                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

                outer_loss += F.cross_entropy(test_logit, test_target)
                outer_loss += tpn_loss


            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            loss_logs.append(outer_loss.item())
            accuracy_logs.append(accuracy.item())

            tpn_acc.div_(args.batch_size)
            tpn_acc_log.append(tpn_acc.item())

            pic_num.div_(args.batch_size)
            fake_pic_num.append(pic_num.item())

            if mode == 'meta_train':
                tpn.zero_grad()
                model.zero_grad()
                outer_loss.backward()
                meta_optimizer.step()

            postfix = {'mode': mode, 'acc': round(np.mean(accuracy_logs).item(), 5), 'pic_num':round(np.mean(fake_pic_num).item(), 5), 't_acc':round(np.mean(tpn_acc_log).item(), 5)}
            pbar.set_postfix(postfix)

    # Save model
    if args.meta_train and args.device == 'cuda' and torch.cuda.is_available():
        filename = os.path.join(args.output_folder, args.save_dir, 'models',
                                'epochs_{}.pt'.format((iteration + 1) * total))
        filename_tpn = os.path.join(args.output_folder, args.save_dir, 'models',
                                'tpn_epochs_{}.pt'.format((iteration + 1) * total))
        if (iteration + 1) * total % 200 == 0:
            if args.device == 'cuda' and torch.cuda.is_available():
                with open(filename, 'wb') as f:
                    state_dict = model.state_dict()
                    torch.save(state_dict, f)
                with open(filename_tpn, 'wb') as f:
                    state_dict = tpn.state_dict()
                    torch.save(state_dict, f)

    return loss_logs, accuracy_logs


def _set_seed(seed=1):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    _set_seed(1)

    import argparse

    parser = argparse.ArgumentParser('LPAP')

    parser.add_argument('--dataset', type=str,
                        help='Dataset: omniglot, miniimagenet, tieredimagenet, cifar_fs, FC100, cub, doublemnist, triplemnist')
    parser.add_argument('--device', type=str, default='cuda', help='gpu device')
    parser.add_argument('--device_index', type=str, default='0')
    parser.add_argument('--num_ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num_shots', type=int, default=1,
                        help='Number of examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_querys', type=int, default=15)

    parser.add_argument('--algorithm', type=str, help='Algorithm : MAML or ANIL or BOIL', default='MAML')
    parser.add_argument('--model', type=str, help='Model: 4conv, resnet', default='4conv')

    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--outer_iter', type=int, default=30,
                        help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 300).')
    parser.add_argument('--train_batches', type=int, default=30,
                        help='Number of batches the model is trained over (i.e., validation save steps) (default: 100).')
    parser.add_argument('--valid_batches', type=int, default=25,
                        help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test_batches', type=int, default=250,
                        help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading (default: 1).')

    parser.add_argument('--output_folder', type=str, default='./output_LPAP/',
                        help='Path to the output folder for saving the model (optional).')

    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--pic_num', type = int, default=75)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_index
    args.save_dir = '{}_{}_{}way_{}shot_{}query_{}'.format(
        args.algorithm,
        args.dataset,
        args.num_ways,
        args.num_shots,
        args.num_querys,
        args.model)

    if args.note != "":
        args.save_dir = args.save_dir + '_{}'.format(
            args.note
        )

    if args.device == 'cuda' and torch.cuda.is_available():
        os.makedirs(os.path.join(args.output_folder, args.save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, args.save_dir, 'models'), exist_ok=True)

    print(args.dataset, args.model, args.algorithm, "Device(%s)" % (args.device_index), "Picnum(%s)" % (args.pic_num))


    model = load_model(args)
    tpn = LabelPropagation(args)

    log_pd = pd.DataFrame(np.zeros([args.outer_iter * args.train_batches, 6]),
                          columns=['train_error', 'train_accuracy', 'valid_error', 'valid_accuracy', 'test_error',
                                   'test_accuracy'])
    filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
    if args.device == 'cuda' and torch.cuda.is_available():
        log_pd.to_csv(filename, index=False)

    for iteration in tqdm(range(args.outer_iter)):
        meta_train_loss_logs, meta_train_accuracy_logs = main(mode='meta_train', iteration=iteration)
        meta_valid_loss_logs, meta_valid_accuracy_logs = main(mode='meta_valid', iteration=iteration)
        log_pd['train_error'][
        (iteration) * args.train_batches:(iteration + 1) * args.train_batches] = meta_train_loss_logs
        log_pd['train_accuracy'][
        (iteration) * args.train_batches:(iteration + 1) * args.train_batches] = meta_train_accuracy_logs
        if meta_valid_loss_logs != None:
            log_pd['valid_error'][(iteration) * args.train_batches] = np.mean(meta_valid_loss_logs)
            log_pd['valid_accuracy'][(iteration) * args.train_batches] = np.mean(meta_valid_accuracy_logs)
        filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
        if args.device == 'cuda' and torch.cuda.is_available():
            log_pd.to_csv(filename, index=False)

    meta_test_loss_logs, meta_test_accuracy_logs = main(mode='meta_test', iteration= 100)
    log_pd['test_error'][args.outer_iter * args.train_batches - 1] = np.mean(meta_test_loss_logs)
    log_pd['test_accuracy'][args.outer_iter * args.train_batches - 1] = np.mean(meta_test_accuracy_logs)
    filename = os.path.join(args.output_folder, args.save_dir, 'logs', 'logs.csv')
    if args.device == 'cuda' and torch.cuda.is_available():
        log_pd.to_csv(filename, index=False)

