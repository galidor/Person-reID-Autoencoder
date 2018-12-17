import utils.dataset_loader as dataset_loader
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import utils.evaluate_rerank as evaluate_rerank
import utils.euclidean_distance as euclidean_distance
import utils.re_ranking_feature as re_ranking_feature
import argparse
import os.path as osp
from utils import progress_bar
from tensorboardX import SummaryWriter

#####################################
# TODO: Add initialization for fully connected layers
#####################################

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('Required Arguments')
optional = parser.add_argument_group('Optional Arguments')
required.add_argument('--data_path', type=str, required=True, help='Path to your folder with dataset')

optional.add_argument('--model_path', type=str, required=False, default='/', help='Path where your network will be'
                                                                                  ' stored')
optional.add_argument('--batch_size', type=int, default=16, help='Batch size')
optional.add_argument('--optim_step', type=int, default=20, help='Number of epochs for learning rate decrease')
optional.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
optional.add_argument('--test', action='store_true', help='Test mode only')
optional.add_argument('--normalize', action='store_true', help='Use if you want your data to be normalized')
optional.add_argument('--reranking', action='store_true', help='Use this if you want to use k-reciprocal encoding')
optional.add_argument('--preprocess_dataset', action='store_true', help='Do it only if you have raw CUHK03 .mat file')
optional.add_argument('--epochs', type=int, default=50, help='Total number of epochs')
optional.add_argument('--feature_dim', type=int, default=2048, help='Size of feature vector.')
opt = parser.parse_args()


def imshow(image):
    image = image.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.pause(0.001)


# top1:0.469286 top5:0.668571 top10:0.750000 mAP:0.432923
class ResNet50(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.feature_dim = feature_dim
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.linear = nn.Linear(2048, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        f = self.linear(f)
        # Normalization and communication channel
        f_power = torch.sqrt(torch.sum(torch.pow(f, 2))/self.feature_dim)
        # fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
        f_normalized = torch.div(f, f_power)
        f_noisy = f_normalized + torch.randn(f_normalized.size()).cuda()*torch.Tensor([0.0]).cuda()
        f_denormalized = torch.mul(f_noisy, f_power)
        # print(torch.max(f_denormalized))
        # print(torch.min(f_denormalized))
        # print(torch.max(f))
        # print(torch.min(f))
        # print(f_noisy)
        # print(f_normalized)
        # print(torch.norm(torch.randn(f_normalized.size()), p=2, dim=1))
        # print(torch.sqrt(torch.sum(torch.pow(torch.randn(2048)*torch.Tensor([0.05]), 2))/2048))
        # print(torch.sqrt(torch.sum(torch.pow(f_normalized, 2))/2048))
        # print(torch.sqrt(torch.sum(torch.pow(f, 2))/2048))
        f = f_denormalized
        y = self.classifier(f)
        if self.training:
            return y
        else:
            return f


def train(epoch, net, criterion, optimizer, data_train_loader):
    start_time = time.time()
    running_loss = 0.0
    matches = 0.0
    net.train()
    for i, data in enumerate(data_train_loader, 0):
        # get the inputs
        inputs, labels, camera_id = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.size())
        # exit()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(labels)
        pred = torch.argmax(outputs, dim=1)
        matches = matches + (pred == labels).sum()

        # print statistics
        running_loss += loss.item()

        progress_bar.print_progress(i+1, len(data_train_loader), prefix='Epoch {}:'.format(epoch + 1),
                                    suffix='Iteration: {} Avg. loss: {:.2f} Avg. accuracy: {:.5f}'
                                           ' Elapsed time: {:.2f} s'.format
                                    ((i + 1), running_loss / opt.batch_size / (i+1),
                                    float(matches)/float(opt.batch_size)/(i+1),
                                    time.time() - start_time), decimals=2)

    writer.add_scalar('Loss', running_loss/10/len(data_train_loader), epoch+1)
    writer.add_scalar('Accuracy', float(matches)/float(opt.batch_size)/len(data_train_loader), epoch+1)


def evaluate_new(net, data_query_loader, data_gallery_loader, feature_dim=2048):
    start_time = time.time()
    net.eval()
    query_features = np.empty([len(data_query_loader), feature_dim])
    query_labels = []
    query_cameras = []
    query_paths = []
    print('Extracting gallery features...')
    for i, query in enumerate(data_query_loader, 0):
        inputs, labels, camera_id = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        query_features[i, :] = np.array(features)
        query_labels.append(labels)
        query_cameras.append(camera_id)
        # query_paths.append(path)
    gallery_features = np.empty([len(data_gallery_loader), feature_dim])
    gallery_labels = []
    gallery_cameras = []
    gallery_paths = []
    print('Extracting query features...')
    for i, query in enumerate(data_gallery_loader, 0):
        inputs, labels, camera_id = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        gallery_features[i, :] = np.array(features)
        gallery_labels.append(labels)
        gallery_cameras.append(camera_id)
        # gallery_paths.append(path)
    query_labels, query_cameras = np.array(query_labels), np.array(query_cameras)
    gallery_labels, gallery_cameras = np.array(gallery_labels), np.array(gallery_cameras)
    if opt.reranking:
        re_ranking_distance = re_ranking_feature.re_ranking(query_features, gallery_features, 20, 6, 0.3)
    else:
        re_ranking_distance = euclidean_distance.calculate_distance(query_features, gallery_features)
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0
    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate_rerank.evaluate(re_ranking_distance[i, :], query_labels[i], query_cameras[i], gallery_labels, gallery_cameras)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_labels)
    print('Top1:{} Top5:{} Top10:{} mAP:{}'.format(CMC[0], CMC[4], CMC[9], ap / len(query_labels)))
    print(time.time() - start_time)
    return CMC, ap


if __name__ == '__main__':

    dataset = dataset_loader.CUHK03(opt.data_path, osp.join(opt.data_path, 'img'), preprocess=opt.preprocess_dataset,
                                    preprocess_check=opt.preprocess_dataset)
    if opt.preprocess_dataset:
        print('Preprocessing completed.')
        exit()
    model_index = 1
    while osp.isfile(osp.join(opt.model_path + "ResNet50-{}".format(model_index))):
        print(osp.join(opt.model_path + "ResNet50-{} exists".format(model_index)))
        model_index = model_index + 1
    writer = SummaryWriter('runs/ResNet50-{}'.format(model_index))

    train_transforms_list = [transforms.Resize((256, 128)),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] \
        if opt.normalize else [transforms.Resize((256, 128)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()]

    data_transform_train = transforms.Compose(train_transforms_list)

    test_transforms_list = [transforms.Resize((256, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] \
        if opt.normalize else [transforms.Resize((256, 128)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()]

    data_transform_test = transforms.Compose(test_transforms_list)

    cuhk_data_train = dataset_loader.ImageDataset(dataset=dataset.train, transform=data_transform_train,
                                                  data_path=False)
    cuhk_data_train_loader = torch.utils.data.DataLoader(cuhk_data_train, batch_size=opt.batch_size, shuffle=True)
    cuhk_data_query = dataset_loader.ImageDataset(dataset=dataset.query, transform=data_transform_test, data_path=False)
    cuhk_data_query_loader = torch.utils.data.DataLoader(cuhk_data_query, batch_size=1, shuffle=False)
    cuhk_data_gallery = dataset_loader.ImageDataset(dataset=dataset.gallery, transform=data_transform_test,
                                                    data_path=False)
    cuhk_data_gallery_loader = torch.utils.data.DataLoader(cuhk_data_gallery, batch_size=1, shuffle=False)

    if opt.test:
        net = torch.load(opt.model_path)
        net = net.cuda()
        cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader, feature_dim=opt.feature_dim)
        exit()

    net = ResNet50(num_classes=767, feature_dim=opt.feature_dim)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([
        {'params': net.base.parameters(), 'lr': opt.learning_rate, 'weight_decay': 5e-4},
        {'params': net.linear.parameters(), 'lr': 0.01, 'weight_decay': 5e-3},
        {'params': net.classifier.parameters(), 'lr': 0.01}
    ], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.optim_step, gamma=0.1, last_epoch=-1)
    acc_best = 0
    for epoch in range(opt.epochs):
        scheduler.step()
        train(epoch, net, criterion, optimizer, cuhk_data_train_loader)
        if (epoch+1) % 10 == 0:
            print("Evaluation...")
            cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader, feature_dim=opt.feature_dim)
            if cmc[0] > acc_best:
                torch.save(net, osp.join(opt.model_path, "ResNet50-{}".format(model_index)))
                acc_best = cmc[0]
                print("Best model at epoch {}".format(epoch+1))