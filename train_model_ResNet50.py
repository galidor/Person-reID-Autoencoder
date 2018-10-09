import dataset_loader
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import datetime
import re_ranking_feature
import evaluate_rerank
import euclidean_distance
import argparse
import os.path as osp
import codecs

model_path = "/home/galidor/Documents/msc_project/models/"
log_path = "/home/galidor/Documents/msc_project/log/"
plots_path = "/home/galidor/Documents/msc_project/plots/"

dataset = dataset_loader.CUHK03()

date = datetime.datetime.now()
log_index = 1
model_index = 1
while os.path.isfile(log_path + "log_{}_{}_{}_{}.txt".format(date.day, date.month, date.year, log_index)):
    print(log_path + "log_{}_{}_{}_{}.txt exists".format(date.day, date.month, date.year, log_index))
    log_index = log_index + 1
# f = open(log_path + "log_{}_{}_{}_{}.txt".format(date.day, date.month, date.year, log_index), 'a')
while os.path.isfile(model_path + "ResNet50-{}".format(model_index)):
    print(model_path + "ResNet50-{} exists".format(model_index))
    model_index = model_index + 1


data_transform_train = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transform_test = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cuhk_data_train = dataset_loader.ImageDataset(dataset=dataset.train, transform=data_transform_train, data_path=True)
cuhk_data_train_loader = torch.utils.data.DataLoader(cuhk_data_train, batch_size=1, shuffle=False)
cuhk_data_query = dataset_loader.ImageDataset(dataset=dataset.query, transform=data_transform_test, data_path=True)
cuhk_data_query_loader = torch.utils.data.DataLoader(cuhk_data_query, batch_size=1, shuffle=False)
cuhk_data_gallery = dataset_loader.ImageDataset(dataset=dataset.gallery, transform=data_transform_test, data_path=True)
cuhk_data_gallery_loader = torch.utils.data.DataLoader(cuhk_data_gallery, batch_size=1, shuffle=False)


def imshow(image):
    image = image.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.pause(0.001)

# top1:0.469286 top5:0.668571 top10:0.750000 mAP:0.432923
class ResNet50(nn.Module):
    def __init__(self, num_classes, feature_dim=2048):
        super(ResNet50, self).__init__()
        # bottleneck = torchvision.models.resnet.Bottleneck(feature_dim,)
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        # self.inplanes = 1024
        # self.layer4 = self._make_layer(torchvision.models.resnet.Bottleneck, feature_dim//4, 3, stride=2)
        # self.feature_extraction = nn.Linear(2048, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)
        # x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # f = self.feature_extraction(f)
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
            print('[%d, %5d] loss: %.3f acc: %.3f matches: %d' %
                  (epoch + 1, i + 1, running_loss / 10, float(matches)/10.0/float(data_train_loader.batch_size), matches))
            last_loss = running_loss / 10
            running_loss = 0.0
            matches = 0.0
    f.write("Loss at the end of epoch {} is {}\n".format((epoch+1), last_loss))
    end_time = time.time() - start_time
    print("Epoch time: {:.2f} seconds".format(end_time))


def evaluate(net, data_query_loader, data_gallery_loader):
    start_time = time.time()
    net.eval()
    query_features = []
    query_labels = []
    query_cameras = []
    for i, query in enumerate(data_query_loader, 0):
        inputs, labels, camera_id = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        query_features.append(features)
        query_labels.append(labels)
        query_cameras.append(camera_id)
    gallery_features = []
    gallery_labels = []
    gallery_cameras = []
    for i, query in enumerate(data_gallery_loader, 0):
        inputs, labels, camera_id = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        gallery_features.append(features)
        gallery_labels.append(labels)
        gallery_cameras.append(camera_id)
    distance_matrix = np.zeros((len(query_features), len(gallery_features)))
    for i, q in enumerate(query_features):
        for j, g in enumerate(gallery_features):
            distance = q - g
            distance = torch.pow(distance, 2)
            distance = torch.sum(distance)
            if (query_labels[i] == gallery_labels[j]) and (query_cameras[i] == gallery_cameras[j]):
                distance = float('inf')
            distance_matrix[i, j] = distance
    distance_sorted = np.argsort(distance_matrix, axis=1)
    gallery_labels = np.array(gallery_labels)
    top1 = np.transpose(gallery_labels[distance_sorted][:, 0]) == np.array(query_labels)
    top5 = np.transpose(gallery_labels[distance_sorted][:, 0:5]) == np.tile(np.array(query_labels), (5, 1))
    top10 = np.transpose(gallery_labels[distance_sorted][:, 0:10]) == np.tile(np.array(query_labels), (10, 1))
    top5 = np.sum(top5, axis=0)
    top10 = np.sum(top10, axis=0)
    top1 = np.clip(top1, 0, 1)
    top5 = np.clip(top5, 0, 1)
    top10 = np.clip(top10, 0, 1)
    top1_acc = np.sum(top1).astype(float) / len(query_labels)
    top5_acc = np.sum(top5).astype(float) / len(query_labels)
    top10_acc = np.sum(top10).astype(float) / len(query_labels)
    print("Top 1: {0:2.2f}% \n Top 5: {1:2.2f}% \n Top 10: {2:2.2f}%".format(top1_acc*100, top5_acc*100, top10_acc*100))
    end_time = time.time() - start_time
    print("Evaluation took approximately {:.2f} seconds".format(end_time))
    f.write("Top 1: {0:2.2f}% \n Top 5: {1:2.2f}% \n Top 10: {2:2.2f}%\n".format(top1_acc*100, top5_acc*100, top10_acc*100))
    f.write("Evaluation took approximately {:.2f} seconds\n".format(end_time))
    return query_features, query_labels


def evaluate_new(net, data_query_loader, data_gallery_loader, feature_dim=2048):
    start_time = time.time()
    net.eval()
    query_features = np.empty([len(data_query_loader), feature_dim])
    query_labels = []
    query_cameras = []
    # query_limit = 9
    # gallery_limit = 19
    # query_features = np.empty([10, 2048])
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
        # if i >= query_limit: break
    gallery_features = np.empty([len(data_gallery_loader), feature_dim])
    gallery_labels = []
    gallery_cameras = []
    gallery_paths = []
    # gallery_features = np.empty([20, 2048])
    for i, query in enumerate(data_gallery_loader, 0):
        inputs, labels, camera_id, path = query
        inputs, labels = inputs.cuda(), labels.cuda()
        features = net(inputs)
        features = features.data.cpu()
        gallery_features[i, :] = np.array(features)
        gallery_labels.append(labels)
        gallery_cameras.append(camera_id)
        gallery_paths.append(path)
        # if i >= gallery_limit: break
    query_labels, query_cameras = np.array(query_labels), np.array(query_cameras)
    gallery_labels, gallery_cameras = np.array(gallery_labels), np.array(gallery_cameras)
    # re_ranking_distance = re_ranking_feature.re_ranking(query_features, gallery_features, 20, 6, 0.3)
    re_ranking_distance = euclidean_distance.calculate_distance(query_features, gallery_features)
    # with open(osp.join(plots_path, 'resnet_ranklist.json'), 'w') as file:
    #     json.dump((query_labels.tolist(), query_cameras.tolist(), query_paths, gallery_labels.tolist(), gallery_cameras.tolist(),
    #                gallery_paths, re_ranking_distance.tolist()), file)
    # print(re_ranking_distance.shape)
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
    print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_labels)))
    f.write('top1:%f top5:%f top10:%f mAP:%f\n' % (CMC[0], CMC[4], CMC[9], ap / len(query_labels)))
    print(time.time() - start_time)
    return CMC, ap


if __name__ == '__main__':
    # test = True
    # if test:
    #     import json
    #
    #     net = torch.load(model_path + "ResNet50-22")
    #     net = net.cuda()
    #     cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader)
    #     # cmc = torch.IntTensor(32).zero_()
    #     cmc = cmc.numpy().tolist()
    #     # with open('test.json') as f:
    #     #     cmc = json.load(f)
    #     # print cmc
    #     # cmc = np.array([1, 2, 3, 4, 5]).tolist()
    #     with open(osp.join(plots_path, 'resnet_rerank.json'), 'w') as f:
    #         json.dump(cmc, f)
    #     exit()

    # import json
    # net = torch.load(model_path + "ResNet50-22")
    # net = net.cuda()
    # cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader)
    # print cmc
    # print ap
    # exit()

    import json
    net = torch.load(model_path + "ResNet50-22")
    net = net.cuda()
    features = np.empty((14096, 2048))
    image_names = []
    for i, data in enumerate(cuhk_data_train_loader, 0):
        inputs, labels, camera_id, image_path = data
        outputs = net(inputs.cuda())
        features[i, :] = outputs.data.cpu().numpy()
        image_names.append(str(image_path[0]).split("/")[-1])
        # if i == 100:
        #     break
        if i%100 == 0:
            print i
    print features
    for i, data in enumerate(cuhk_data_query_loader, 7368):
        inputs, labels, camera_id, image_path = data
        outputs = net(inputs.cuda())
        features[i, :] = outputs.data.cpu().numpy()
        image_names.append(str(image_path[0]).split("/")[-1])
        # if i == 100:
        #     break
        if i%100 == 0:
            print i
    print features
    for i, data in enumerate(cuhk_data_gallery_loader, 7368+1400):
        inputs, labels, camera_id, image_path = data
        outputs = net(inputs.cuda())
        features[i, :] = outputs.data.cpu().numpy()
        image_names.append(str(image_path[0]).split("/")[-1])
        # if i == 100:
        #     break
        if i%100 == 0:
            print i
    print features
    with open(osp.join('feature_vectors.json'), 'w') as f:
        # json.dump(features, f)
        json.dump(features.tolist(), f, separators=(',', ':'), sort_keys=True, indent=4)
    with open(osp.join('file_names.json'), 'w') as f:
        json.dump(image_names, f)
    exit()


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feature_dimension', type=int, default=2048)
    parser.add_argument('--optim_step', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--test', action='store_true')
    opt = parser.parse_args()

    if opt.test:
        import json
        net = torch.load(model_path + "ResNet50-22")
        net = net.cuda()
        cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader, feature_dim=opt.feature_dimension)
        cmc = cmc.numpy().tolist()
        with open(osp.join(plots_path, 'resnet_jpeg_rerank.json'), 'w') as f:
            json.dump(cmc, f)
        exit()

    net = ResNet50(num_classes=767, feature_dim=opt.feature_dimension)
    print net.modules
    # net = torch.load(model_path+"ResNet50-1")
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-04)
    optimizer = optim.SGD([
        {'params': net.base.parameters(), 'lr': opt.learning_rate},
        # {'params': net.feature_extraction.parameters(), 'lr': 0.01},
        {'params': net.classifier.parameters(), 'lr': 0.01}
    ], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.optim_step, gamma=0.1, last_epoch=-1)

    # data = next(iter(cuhk_data_query_loader))
    # inputs, labels, camera_id = data
    # net.eval()
    # features = net(inputs.cuda())
    acc_best = 0
    # evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader, feature_dim=opt.feature_dimension)
    for epoch in range(80):  # loop over the dataset multiple times
        scheduler.step()
        train(epoch, net, criterion, optimizer, cuhk_data_train_loader)
        # f.write("Epoch {}\n".format(epoch))
        if (epoch+1) % 10 == 0:
            # f.write("Evaluation...\n")
            cmc, ap = evaluate_new(net, cuhk_data_query_loader, cuhk_data_gallery_loader, feature_dim=opt.feature_dimension)
            if cmc[0] > acc_best:
                torch.save(net, model_path + "ResNet50-{}".format(model_index))
                acc_best = cmc[0]
                f.write("Best model at epoch {}\n".format(epoch+1))
    f.close()

