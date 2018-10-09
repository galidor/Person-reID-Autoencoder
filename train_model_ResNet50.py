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
import os.path
import utils.evaluate_rerank as evaluate_rerank
import utils.euclidean_distance as euclidean_distance
import utils.re_ranking_feature as re_ranking_feature
import argparse
import os.path as osp

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--data_path', type=str, required=True)
required.add_argument('--model_path', type=str, required=True)
optional.add_argument('--batch_size', type=int, default=16)
optional.add_argument('--optim_step', type=int, default=20)
optional.add_argument('--learning_rate', type=float, default=0.01)
optional.add_argument('--test', action='store_true', help='Test mode only')
optional.add_argument('--normalize', action='store_true')
optional.add_argument('--reranking', action='store_true', help='Use this if you want to use k-reciprocal encoding')
optional.add_argument('--preprocess_dataset', action='store_true', help='Do it only if you have raw CUHK03 .mat file.')
optional.add_argument('--epochs', type=int, default=50)
opt = parser.parse_args()

dataset = dataset_loader.CUHK03(opt.data_path, osp.join(opt.data_path, 'img'), preprocess=opt.preprocess_dataset,
                                preprocess_check=opt.preprocess_dataset)
model_index = 1
while os.path.isfile(osp.model_path + "ResNet50-{}".format(model_index)):
    print(osp.model_path + "ResNet50-{} exists".format(model_index))
    model_index = model_index + 1

train_transforms_list = [transforms.Resize((256, 128)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]\
    if opt.normalize else [transforms.Resize((256, 128)),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]

data_transform_train = transforms.Compose(train_transforms_list)

test_transforms_list = [transforms.Resize((256, 128)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]\
    if opt.normalize else [transforms.Resize((256, 128)),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]

data_transform_test = transforms.Compose(test_transforms_list)

cuhk_data_train = dataset_loader.ImageDataset(dataset=dataset.train, transform=data_transform_train, data_path=False)
cuhk_data_train_loader = torch.utils.data.DataLoader(cuhk_data_train, batch_size=opt.batch_size, shuffle=True)
cuhk_data_query = dataset_loader.ImageDataset(dataset=dataset.query, transform=data_transform_test, data_path=False)
cuhk_data_query_loader = torch.utils.data.DataLoader(cuhk_data_query, batch_size=1, shuffle=False)
cuhk_data_gallery = dataset_loader.ImageDataset(dataset=dataset.gallery, transform=data_transform_test, data_path=False)
cuhk_data_gallery_loader = torch.utils.data.DataLoader(cuhk_data_gallery, batch_size=1, shuffle=False)


def imshow(image):
    image = image.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.pause(0.001)

# top1:0.469286 top5:0.668571 top10:0.750000 mAP:0.432923
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(labels)
        pred = torch.argmax(outputs, dim=1)
        matches = matches + (pred == labels).sum()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:
            print('Epoch: {} Iterations: {} Avg. loss: {:.2f} Matches: {}'.format
                  (epoch + 1, i + 1, running_loss / 10, float(matches)/10.0/float(data_train_loader.batch_size), matches))
            last_loss = running_loss / 10
            running_loss = 0.0
            matches = 0.0
    end_time = time.time() - start_time
    print("Epoch time: {:.2f} seconds".format(end_time))


def evaluate_new(net, data_query_loader, data_gallery_loader, feature_dim=2048):
    start_time = time.time()
    net.eval()
    query_features = np.empty([len(data_query_loader), feature_dim])
    query_labels = []
    query_cameras = []
    query_paths = []
    for i, query in enumerate(data_query_loader, 0):
        inputs, labels, camera_id, path = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        query_features[i, :] = np.array(features)
        query_labels.append(labels)
        query_cameras.append(camera_id)
        query_paths.append(path)
    gallery_features = np.empty([len(data_gallery_loader), feature_dim])
    gallery_labels = []
    gallery_cameras = []
    gallery_paths = []
    for i, query in enumerate(data_gallery_loader, 0):
        inputs, labels, camera_id, path = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        gallery_features[i, :] = np.array(features)
        gallery_labels.append(labels)
        gallery_cameras.append(camera_id)
        gallery_paths.append(path)
    query_labels, query_cameras = np.array(query_labels), np.array(query_cameras)
    gallery_labels, gallery_cameras = np.array(gallery_labels), np.array(gallery_cameras)
    if opt.rerank:
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
    print('Top1:{} Top5:{} Top10:%f mAP:{}'.format(CMC[0], CMC[4], CMC[9], ap / len(query_labels)))
    print(time.time() - start_time)
    return CMC, ap


if __name__ == '__main__':

    if opt.test:
        net = torch.load(opt.model_path)
        net = net.cuda()
        cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader)
        exit()

    net = ResNet50(num_classes=767)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([
        {'params': net.base.parameters(), 'lr': opt.learning_rate},
        {'params': net.classifier.parameters(), 'lr': 0.01}
    ], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.optim_step, gamma=0.1, last_epoch=-1)

    acc_best = 0
    for epoch in range(opt.epochs):
        scheduler.step()
        train(epoch, net, criterion, optimizer, cuhk_data_train_loader)
        if (epoch+1) % 10 == 0:
            print("Evaluation...")
            cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader)
            if cmc[0] > acc_best:
                torch.save(net, osp.join(opt.model_path, "ResNet50-{}".format(model_index)))
                acc_best = cmc[0]
                print("Best model at epoch {}".format(epoch+1))


