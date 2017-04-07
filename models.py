import torch
import torch.nn as nn
import torch.nn.parallel


class CNN(nn.Module):
    def __init__(self, nc, input_size, hparams, ngpu=1, leaky_slope=0.01, std=0.01):
        super(CNN, self).__init__()
        self.ngpu = ngpu  # num of gpu's to use
        self.leaky_slope = leaky_slope  # slope for leaky_relu activation
        self.std = std  # standard deviation for weights initialization
        self.input_size = input_size  # expected input size

        main = nn.Sequential()
        in_feat, num = nc, 0
        for op, k, s, out_feat, bn, dp, h in hparams:
            # add operation: conv2d or convTranspose2d
            if op == 'conv2d':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.conv'.format(num, in_feat, out_feat),
                    nn.Conv2d(in_feat, out_feat, k, s, 0, bias=False))
            elif op == 'convt2d':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.convt'.format(num,in_feat, out_feat),
                    nn.ConvTranspose2d(in_feat, out_feat, k, s, 0, bias=False))
            else:
                raise Exception('Not supported operation: {0}'.format(op))
            num += 1
            # add batch normalization layer
            if bn:
                main.add_module(
                    '{0}.pyramid.{1}-{2}.batchnorm'.format(num, in_feat, out_feat),
                    nn.BatchNorm2d(out_feat))
                num += 1
            # add dropout layer
            main.add_module(
                '{0}.pyramid.{1}-{2}.dropout'.format(num, in_feat, out_feat),
                nn.Dropout2d(p=dp))
            num += 1
            # add activation
            if h == 'leaky_relu':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.leaky_relu'.format(num, in_feat, out_feat),
                    nn.LeakyReLU(self.leaky_slope, inplace=True))
            elif h == 'sigmoid':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.sigmoid'.format(num, in_feat, out_feat),
                    nn.Sigmoid())
            elif h == 'maxout':
                # TODO: implement me
                # https://github.com/IshmaelBelghazi/ALI/blob/master/ali/bricks.py#L338-L380
                raise NotImplementedError('Maxout is not implemented.')
            elif h == 'relu':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.relu'.format(num, in_feat, out_feat),
                    nn.ReLU(inplace=True))
            elif h == 'tanh':
                main.add_module(
                    '{0}.pyramid.{1}-{2}.tanh'.format(num, in_feat, out_feat),
                    nn.Tanh())
            elif h == 'linear':
                num -= 1  # 'Linear' do nothing
            else:
                raise Exception('Not supported activation: {0}'.format(h))
            num += 1
            in_feat = out_feat
        self.main = main

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, self.std)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, self.std)
                m.bias.data.zero_()

    def forward(self, input):
        assert input.size(2) == self.input_size,\
            'Wrong input size: {0}. Expected {1}'.format(input.size(2),
                                                         self.input_size)
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            gpu_ids = range(self.ngpu)
            output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        else:
            output = self.main(input)
        return output


def create_svhn_gz(nz=256, ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 5, 1,   32,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 2,   64,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1,  128,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 2,  256,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1,  512,  True, 0.0, 'leaky_relu'],
        ['conv2d', 1, 1,  512,  True, 0.0, 'leaky_relu'],
        ['conv2d', 1, 1, 2*nz, False, 0.0, 'linear'],
    ]
    return CNN(3, 32, hparams, ngpu)


def create_svhn_gx(nz=256, ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['convt2d', 4, 1, 256,  True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 2, 128,  True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 1,  64,  True, 0.0, 'leaky_relu'],
        ['convt2d', 4, 2,  32,  True, 0.0, 'leaky_relu'],
        ['convt2d', 5, 1,  32,  True, 0.0, 'leaky_relu'],
        ['convt2d', 1, 1,  32,  True, 0.0, 'leaky_relu'],
        ['conv2d',  1, 1,   3, False, 0.0, 'sigmoid'],
    ]
    return CNN(nz, 1, hparams, ngpu)


def create_svhn_dx(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 5, 1,  32, False, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2,  64,  True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 1, 128,  True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 2, 256,  True, 0.2, 'leaky_relu'],
        ['conv2d', 4, 1, 512,  True, 0.2, 'leaky_relu'],
    ]
    return CNN(3, 32, hparams, ngpu)


def create_svhn_dz(nz=256, ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 1, 1, 512, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 512, False, 0.2, 'leaky_relu'],
    ]
    return CNN(nz, 1, hparams, ngpu)


def create_svhn_dxz(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 1, 1, 1024, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 1024, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1,    1, False, 0.2, 'sigmoid'],
    ]
    return CNN(1024, 1, hparams, ngpu)


def create_celeba_gz(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 2, 1,  64,  True, 0.0, 'leaky_relu'],
        ['conv2d', 7, 2, 128,  True, 0.0, 'leaky_relu'],
        ['conv2d', 5, 2, 256,  True, 0.0, 'leaky_relu'],
        ['conv2d', 7, 2, 256,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1, 512,  True, 0.0, 'leaky_relu'],
        ['conv2d', 1, 1, 512, False, 0.0, 'linear'],
    ]
    return CNN(3, 64, hparams, ngpu)


def create_celeba_gx(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // BN // dropout // nonlinearity
        ['convt2d', 4, 1, 512,  True, 0.0, 'leaky_relu'],
        ['convt2d', 7, 2, 256,  True, 0.0, 'leaky_relu'],
        ['convt2d', 5, 2, 256,  True, 0.0, 'leaky_relu'],
        ['convt2d', 7, 2, 128,  True, 0.0, 'leaky_relu'],
        ['convt2d', 2, 1,  64,  True, 0.0, 'leaky_relu'],
        ['conv2d',  1, 1,   3, False, 0.0, 'sigmoid'],
    ]
    return CNN(512, 1, hparams, ngpu)


def create_celeba_dx(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 2, 1,  64,  True, 0.0, 'leaky_relu'],
        ['conv2d', 7, 2, 128,  True, 0.0, 'leaky_relu'],
        ['conv2d', 5, 2, 256,  True, 0.0, 'leaky_relu'],
        ['conv2d', 7, 2, 256,  True, 0.0, 'leaky_relu'],
        ['conv2d', 4, 1, 512,  True, 0.0, 'leaky_relu'],
    ]
    return CNN(3, 64, hparams, ngpu)


def create_celeba_dz(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 1, 1, 1024, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 1024, False, 0.2, 'leaky_relu'],
    ]
    return CNN(512, 1, hparams, ngpu)


def create_celeba_dxz(ngpu=1):
    hparams = [
        # op // kernel // strides // fmaps // batch_norm // dropout // nonlinearity
        ['conv2d', 1, 1, 2048, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1, 2048, False, 0.2, 'leaky_relu'],
        ['conv2d', 1, 1,    1, False, 0.2, 'sigmoid'],
    ]
    return CNN(1536, 1, hparams, ngpu)


def create_models(dataset, nz, ngpu=1):
    if dataset == 'cifar10':
        raise NotImplementedError('Cifar10 needs Maxout which is not implemented yet.')
    elif dataset == 'svhn':
        gx = create_svhn_gx(nz, ngpu)
        gz = create_svhn_gz(nz, ngpu)
        dx = create_svhn_dx(ngpu)
        dz = create_svhn_dz(nz, ngpu)
        dxz = create_svhn_dxz(ngpu)
    elif dataset == 'celeba':
        gx = create_celeba_gx(ngpu)
        gz = create_celeba_gz(ngpu)
        dx = create_celeba_dx(ngpu)
        dz = create_celeba_dz(ngpu)
        dxz = create_celeba_dxz(ngpu)
    else:
        raise Exception('Not supported dataset: {0}'.format(dataset))
    return gx, gz, dx, dz, dxz

