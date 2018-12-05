import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
from utils import dataset_loader
import numpy as np
import torch.nn.functional as F
from PIL import Image
import os
import datetime
# import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os.path as osp


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
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
optional.add_argument('--adversarial', action='store_true')
optional.add_argument('--mse', action='store_true')
optional.add_argument('--feat', action='store_true')
optional.add_argument('--feature_dimension', type=int, default=2048)

opt = parser.parse_args()


model_path = "/home/galidor/Documents/msc_project/models/"
log_path = "/home/galidor/Documents/msc_project/log/"
data_path = "/home/galidor/Documents/msc_project/datasets/"

date = datetime.datetime.now()
dataset = dataset_loader.CUHK03(opt.data_path, osp.join(opt.data_path, 'img/'))

data_transform_train = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transform_trainR = transforms.Compose([
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

transform_inv = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                         std=[1/0.229, 1/0.224, 1/0.225])
])

cuhk_data_train = dataset_loader.ImageDataset(dataset=dataset.train, transform=data_transform_train)
cuhk_data_train_loader = torch.utils.data.DataLoader(cuhk_data_train, batch_size=16, shuffle=True)
cuhk_data_query = dataset_loader.ImageDataset(dataset=dataset.query, transform=data_transform_test)
cuhk_data_query_loader = torch.utils.data.DataLoader(cuhk_data_query, batch_size=1, shuffle=True)
cuhk_data_gallery = dataset_loader.ImageDataset(dataset=dataset.gallery, transform=data_transform_test)
cuhk_data_gallery_loader = torch.utils.data.DataLoader(cuhk_data_gallery, batch_size=1, shuffle=True)


model_index = 1
while os.path.isfile(model_path + "Encoder-{}".format(model_index)):
    print(model_path + "Encoder-{} exists".format(model_index))
    model_index = model_index + 1


def img2writer(image, name):
    # image = image.numpy().transpose((1, 2, 0))
    # image = np.clip(image, 0, 1)
    # im = Image.fromarray(np.uint8(image * 255), 'RGB')
    writer.add_image('Images/' + name, image)


feature_dim = 512
ngf = 64


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(opt.feature_dimension, 512, 2, 1, 0, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # 2x2x512
        self.deconv2 = nn.ConvTranspose2d(512, 512, 4, 2, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm2d(512)
        # 4x4x512
        self.deconv3 = nn.ConvTranspose2d(512, 512, 4, 2, padding=1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm2d(512)
        # 8x8x512
        self.deconv4 = nn.ConvTranspose2d(512, 512, 4, 2, padding=1, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        # 16x16x512
        self.deconv5 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5 = nn.BatchNorm2d(256)
        # 32x32x256
        self.deconv6 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6 = nn.BatchNorm2d(128)
        # 64x64x128
        self.deconv7 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7 = nn.BatchNorm2d(64)
        # 128x128x64
        self.deconv8 = nn.ConvTranspose2d(64, 3, 3, (2, 1), padding=(1, 1), output_padding=(1, 0), bias=False)
        self.relu8 = nn.ReLU(inplace=True)
        # output: 256x128x3

    def forward(self, x):

        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.deconv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.deconv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.deconv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.deconv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.deconv7(x)
        x = self.relu7(x)
        x = self.bn7(x)
        print(x.shape)
        x = self.deconv8(x)
        output = self.relu8(x)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.padding1 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv1 = nn.Conv2d(3, 64, 3, (2, 1), 0, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        # 128x128x64
        self.padding2 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 0, bias=False)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm2d(128)
        # 64x64x128
        self.padding3 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 0, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm2d(256)
        # 32x32x256
        self.padding4 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 0, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4 = nn.BatchNorm2d(512)
        # 16x16x512
        self.padding5 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 0, bias=False)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5 = nn.BatchNorm2d(512)
        # 8x8x512
        self.padding6 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 0, bias=False)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6 = nn.BatchNorm2d(512)
        # 4x4x512
        self.padding7 = nn.ReplicationPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(512, 512, 4, 2, 0, bias=False)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7 = nn.BatchNorm2d(512)
        # 2x2x512
        self.conv8 = nn.Conv2d(512, opt.feature_dimension, 2, 1, 0, bias=False)
        self.relu8 = nn.LeakyReLU(0.2, inplace=True)
        # 1x1xfeature_dimension

    def forward(self, x):
        x = self.padding1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        print(x.shape)
        x = self.padding2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.padding3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.padding4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.padding5(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.padding6(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn6(x)
        x = self.padding7(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        output = self.relu8(x)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, (4, 2), 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self. conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self. relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(64 * 8, 1, 8, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        output = self.sigmoid(x)
        return output.view(-1, 1).squeeze(1)


class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, (4, 2), 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 32 x 32
        self. conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self. relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3 = nn.BatchNorm2d(64 * 4)
        # state size. (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4 = nn.BatchNorm2d(64 * 8)
        # state size. (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(64 * 8, 1, 8, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.conv5(x)
        output = self.sigmoid(x)
        return output.view(-1, 1).squeeze(1)


class ResNet50(nn.Module):
    def __init__(self, num_classes, feature_dim=2048):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
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


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


netE = Encoder().cuda()
netD = Decoder().cuda()
sample_data = next(iter(cuhk_data_gallery_loader))
sample_image, _, _ = sample_data
print(sample_image.shape)
sample_image = sample_image.cuda()
sample_output = netE(sample_image)
print(sample_output.shape)
sample_reovered = netD(sample_output)
print(sample_reovered.shape)
exit()

netF = torch.load(model_path + "ResNet50-22").cuda()
netE = Encoder().cuda()
netE.apply(weights_init)
netD = Decoder().cuda()
netD.apply(weights_init)
criterion = nn.MSELoss().cuda()
# criterion = nn.L1Loss().cuda()

optimizerAE = optim.Adam([{'params': netE.parameters()},
                         {'params': netD.parameters()}], lr=0.0002, betas=(0.5, 0.999))

netE.train()
netD.train()
running_loss = []
running_loss2 = []

writer_name = opt.name
writer = SummaryWriter('runs/' + writer_name)

netE.eval()
netD.eval()
sample_photo = next(iter(cuhk_data_gallery_loader))
img, _, _ = sample_photo
img = img.cuda()
imgname = 'test{}test.png'.format(32)
photo_recovered = netD(netE(img))
img2writer(photo_recovered[0].data.cpu(), 'test')


for epoch in range(150):
    print("Epoch {}".format(epoch+1))
    netE.train()
    netD.train()
    netF.eval()
    for i, data in enumerate(cuhk_data_train_loader, 0):
        inputs, labels, cam_id = data
        inputs = inputs.cuda()
        optimizerAE.zero_grad()
        codes = netE(inputs)
        outputs = netD(codes)
        features_inputs = netF(inputs)
        features_outputs = netF(outputs)
        loss2 = F.mse_loss(features_inputs, features_outputs) * 10
        loss1 = criterion(outputs, inputs) * 50
        loss = loss1 + loss2
        # loss2 = Variable(loss2, requires_grad=True)
        loss.backward(retain_graph=False)
        optimizerAE.step()

        running_loss.append(loss1.item())
        running_loss2.append(loss2)
        if (i+1)%10 == 0:
            print('L1: {}   Sum: {}'.format(loss1, loss))
    running_loss = sum(running_loss) / len(running_loss) / 50.0
    running_loss2 = sum(running_loss2) / len(running_loss2) / 10.0
    print("[{} {}] AE Sim Loss: {}, AE Feat Loss: {}".format(epoch+1, i+1, running_loss, running_loss2))
    writer.add_scalar('/MSE Feat Loss', running_loss2, epoch+1)
    writer.add_scalar('/MSE Loss', running_loss, epoch+1)
    running_loss = []
    running_loss2 = []
    # if (epoch+1) % 5 == 0:
    # if epoch > 100:
    netE.eval()
    netD.eval()
    sample_photo = next(iter(cuhk_data_gallery_loader))
    img, _, _ = sample_photo
    img = img.cuda()
    photo_recovered = netD(netE(img))
    # img = transform_inv(img.cpu().data[0])
    # photo_recovered = transform_inv(photo_recovered.cpu().data[0])
    if opt.mse:
        imgname = 'epoch{}mse'.format(epoch)
    elif opt.adversarial:
        imgname = 'epoch{}adv'.format(epoch)
    else:
        imgname = 'epoch{}vanilla'.format(epoch)
    img2writer(img.cpu().data[0], 'real/{}'.format(imgname))
    img2writer(photo_recovered.cpu().data[0], 'rec/{}'.format(imgname))
# f.close()
torch.save(netD, model_path + "Decoder-{}".format(model_index))
torch.save(netE, model_path + "Encoder-{}".format(model_index))
writer.close()
    # imshow(img)
    # imshow(photo_recovered)
