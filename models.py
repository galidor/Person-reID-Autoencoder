from torch import nn
import torch
import math


class UniversalAutoencoder(nn.Module):
    def __init__(self, im_size, max_depth):
        super(UniversalAutoencoder, self).__init__()
        im_height, im_width, im_depth = im_size
        base_depth = 64
        depth_mult = 1
        kernel_size = 3
        original_im_width = im_width
        original_im_height = im_height
        original_im_depth = im_depth

        # Encoder sequential generation
        layers = [nn.ReplicationPad2d((math.floor(kernel_size/2), math.floor(kernel_size/2),
                                       math.floor(kernel_size / 2), math.floor(kernel_size/2))),
                  nn.Conv2d(im_depth, base_depth, kernel_size, 2),
                  nn.ReLU(inplace=True)]
        im_width = math.floor((im_width + 2*math.floor(kernel_size/2)-kernel_size)/2 + 1)
        im_height = math.floor((im_height + 2 * math.floor(kernel_size / 2) - kernel_size) / 2 + 1)
        im_depth = base_depth*depth_mult
        while im_width != 2:
            if im_depth < max_depth:
                depth_mult *= 2
            layers.extend([nn.ReplicationPad2d((math.floor(kernel_size/2), math.floor(kernel_size/2),
                                                math.floor(kernel_size / 2), math.floor(kernel_size/2))),
                           nn.Conv2d(im_depth, base_depth*depth_mult, kernel_size, 2),
                           nn.BatchNorm2d(base_depth*depth_mult),
                           nn.ReLU(inplace=True)])
            im_width = math.floor((im_width + 2 * math.floor(kernel_size / 2) - kernel_size) / 2 + 1)
            im_height = math.floor((im_height + 2 * math.floor(kernel_size / 2) - kernel_size) / 2 + 1)
            im_depth = base_depth * depth_mult
        layers.extend([nn.ReplicationPad2d((math.floor(kernel_size / 2), math.floor(kernel_size / 2),
                                            math.floor(kernel_size / 2), math.floor(kernel_size / 2))),
                       nn.Conv2d(im_depth, base_depth * depth_mult, kernel_size, 2),
                       nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*layers)

        #Decoder sequential generation
        layers_num = round(math.log(original_im_width, 2))
        channel_num = [original_im_depth, base_depth]
        channel_num.extend([base_depth*i for i in (2**j for j in range(1, layers_num))])
        channel_num = [min(i, max_depth) for i in channel_num]
        print(channel_num)
        channel_num = channel_num[::-1]
        layers = []
        for i in range(len(channel_num)-2):
            layers.extend([nn.Upsample(scale_factor=2),
                           nn.ReplicationPad2d((math.floor(kernel_size / 2), math.floor(kernel_size / 2),
                                                math.floor(kernel_size / 2), math.floor(kernel_size / 2))),
                           nn.Conv2d(channel_num[i], channel_num[i+1], kernel_size, 1),
                           nn.BatchNorm2d(channel_num[i+1]),
                           nn.LeakyReLU(inplace=True, negative_slope=0.2)])
        layers.extend([nn.Upsample(scale_factor=2),
                       nn.ReplicationPad2d((math.floor(kernel_size / 2), math.floor(kernel_size / 2),
                                            math.floor(kernel_size / 2), math.floor(kernel_size / 2))),
                       nn.Conv2d(channel_num[-2], channel_num[-1], kernel_size, 1),
                       nn.Tanh()])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = UniversalAutoencoder((512, 512, 3), 512)
noise = torch.randn(1, 3, 512, 512)
print(net)