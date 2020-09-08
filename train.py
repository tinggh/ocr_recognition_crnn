from __future__ import division, print_function

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
from torch.autograd import Variable

import dataset
import keys, data_dict
import utils
from models import crnn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/yjwang/Workspaces/data/recognition/images/', help='path to dataset')
    parser.add_argument('--train_label_path', default='/home/yjwang/Workspaces/data/recognition/train.txt', help='path to dataset')
    parser.add_argument('--val_label_path', default='/home/yjwang/Workspaces/data/recognition/test.txt', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--cuda', default='0', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrained', default='expr/denseNet_4_0.9664375.pth', help="path to pretrained model (to continue training)")
    parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--saveAccuracy', type=int, default=0.5, help='Interval to be displayed')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
    parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    print(opt)
    return opt


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def val(crnn, valid_loader, criterion, max_iter=1000):
    print('Start val')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    
    val_iter = iter(valid_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(valid_loader))
    for i in range(max_iter):
        names, images, texts = val_iter.next()
        batch_size = images.size(0)
        t, l = converter.encode(texts)
        images = images.cuda()
        preds = crnn(images)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, t, preds_size, l) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, texts):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for name, raw_pred, pred, gt in zip(names, raw_preds, sim_preds, texts):
        print('%-20s:%-20s => %-20s, gt: %-20s' % (name, raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    return accuracy



def train(crnn, train_loader, criterion, optimizer, valid_loader):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    train_iter = iter(train_loader)
    # loss averager
    loss_avg = utils.averager()
    for i in range(len(train_loader)):
        data = train_iter.next()
        _, images, texts = data
        batch_size = images.size(0)
        t, l = converter.encode(texts)
        images = images.cuda()
        preds = crnn(images)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, t, preds_size, l) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)
        if (i+1) % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()
        # if (i+1) % opt.valInterval:
        #     _ = val(crnn, valid_loader, criterion, max_iter=100)
        #     for p in crnn.parameters():
        #         p.requires_grad = True
        #     crnn.train()


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0

if __name__ == "__main__":

    opt = get_args()

    if not os.path.exists(opt.expr_dir):
        os.makedirs(opt.expr_dir)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_dataset = dataset.folderDataset(opt.root, opt.train_label_path, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
    assert train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True, num_workers=int(opt.workers))

    valid_dataset = dataset.folderDataset(
        opt.root, opt.val_label_path, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))

    alphabet = data_dict.alphabet
    nclass = len(alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(alphabet)
    criterion = torch.nn.CTCLoss(reduction='sum')
    ## ResNet18
    # crnn = crnn.CRNN(CNN="ResNet18", RNN='DoubleBiLSTM',  nIn=512, n_class=nclass, nHidden=256, nLayer=1, dropout=0)
    # Densenet18
    CNN = "denseNet"
    RNN = 'DoubleBiLSTM'
    crnn = crnn.CRNN(CNN=CNN, RNN=RNN,  nIn=128, n_class=nclass, nHidden=256, nLayer=1, dropout=0)
    # CNN = "DenseNet18_256"
    # RNN = "DoubleBiLSTM"
    # crnn = crnn.CRNN(CNN=CNN, RNN=RNN,  nIn=256, n_class=nclass, nHidden=256, nLayer=1, dropout=0)

    crnn.apply(weights_init)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        crnn.load_state_dict(torch.load(opt.pretrained))

    from torchsummary import summary
    summary(crnn, (1, 32, 280))


    if opt.cuda:
        crnn = crnn.cuda()
        # crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    crnn.register_backward_hook(backward_hook)

    for epoch in range(opt.nepoch):
        train(crnn, train_loader, criterion, optimizer, valid_loader)
        acc = val(crnn, valid_loader, criterion)
        if acc > opt.saveAccuracy:
            torch.save(
                crnn.state_dict(), '{0}/{1}_{2}_{3}.pth'.format(opt.expr_dir,CNN, epoch, acc))
