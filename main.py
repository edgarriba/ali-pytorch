from __future__ import print_function
import os
import argparse
from itertools import chain

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import logger
import numpy as np
import models as ali

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn | celeba')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch-size', type=int, default=100, help='input batch size')
parser.add_argument('--image-size', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--netGx', default='', help="path to netGx (to continue training)")
parser.add_argument('--netGz', default='', help="path to netGz (to continue training)")
parser.add_argument('--netDz', default='', help="path to netDz (to continue training)")
parser.add_argument('--netDx', default='', help="path to netDx (to continue training)")
parser.add_argument('--netDxz', default='', help="path to netDxz (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
opt = parser.parse_args()
print(opt)

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

# opt.manualSeed = random.randint(1, 10000) # fix seed
opt.seed = 0
print("Random Seed: ", opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(opt.seed)

# create logger
LOG_DIR = '{0}/logger'.format(opt.experiment)

# some hyperparameters we wish to save for this experiment
hyperparameters = dict(regularization=1, n_epochs=opt.epochs)
# options for the remote visualization backend
visdom_opts = dict(server='http://localhost', port=8097)
# create logger for visdom
xp = logger.Experiment('xp_name', use_visdom=True, visdom_opts=visdom_opts)
# log the hyperparameters of the experiment
xp.log_config(hyperparameters)
# create parent metric for training metrics (easier interface)
train_metrics = xp.ParentWrapper(tag='train', name='parent',
                                 children=(xp.AvgMetric(name='loss1'),
                                           xp.AvgMetric(name='loss2')))

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# setup transformations
transforms = transforms.Compose([
    transforms.Scale(opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms)
elif opt.dataset == 'svhn':
    dataset = dset.SVHN(root=opt.dataroot, download=True,
                        transform=transforms)
elif opt.dataset == 'celebA':
    raise NotImplementedError('CelebA is not available in visiontorch yet ;D')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)  # number of GPUs
nc = int(opt.nc)  # number input channels
nz = int(opt.nz)  # latent space size

eps = 1e-15  # to avoid possible NaN/Inf during backward

# create models and load parameters if needed
netGx, netGz, netDx, netDz, netDxz = ali.create_models(opt.dataset, nz, ngpu)

if opt.netGz != '':  # load checkpoint if needed
    netGz.load_state_dict(torch.load(opt.netGz))
print(netGz)

if opt.netGx != '':  # load checkpoint if needed
    netGx.load_state_dict(torch.load(opt.netGx))
print(netGx)

if opt.netDz != '':  # load checkpoint if needed
    netDz.load_state_dict(torch.load(opt.netDz))
print(netDz)

if opt.netDx != '':  # load checkpoint if needed
    netDx.load_state_dict(torch.load(opt.netDx))
print(netDx)

if opt.netDxz != '':  # load checkpoint if needed
    netDxz.load_state_dict(torch.load(opt.netDxz))
print(netDxz)

# setup input tensors
x = torch.FloatTensor(opt.batch_size, nc, opt.image_size, opt.image_size)
z = torch.FloatTensor(opt.batch_size, nz, 1, 1)

if opt.cuda:
    netGx.cuda(), netGz.cuda()
    netDx.cuda(), netDz.cuda(), netDxz.cuda()
    x, z = x.cuda(), z.cuda()

x, z = Variable(x), Variable(z)

# setup optimizer
dis_params = chain(netDx.parameters(), netDz.parameters(), netDxz.parameters())
gen_params = chain(netGx.parameters(), netGz.parameters())

kwargs_adam = {'lr': opt.lr, 'betas': (opt.beta1, opt.beta2)}
optimizerD = optim.Adam(dis_params, **kwargs_adam)
optimizerG = optim.Adam(gen_params, **kwargs_adam)


def train(dataloader, epoch):
    # Set the networks in train mode (apply dropout when needed)
    netDx.train(), netDz.train(), netDxz.train()
    netGx.train(), netGz.train()

    for batch_id, (real_cpu, _) in enumerate(dataloader):
        ###########################
        # Prepare data
        ###########################
        batch_size = real_cpu.size(0)
        x.data.resize_(real_cpu.size()).copy_(real_cpu)

        # clamp parameters to a cube
        for p in netDx.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
        for p in netDz.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
        for p in netDxz.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        # init gradients
        netDz.zero_grad(), netDx.zero_grad(), netDxz.zero_grad()
        # netGx.zero_grad(), netGz.zero_grad()

        # generate random data
        z.data.resize_(batch_size, nz, 1, 1)
        z.data.normal_(0, 1)
        epsilon = np.random.normal(0, 1)

        # equation (2) from the paper
        # q(z | x) = N(mu(x), sigma^2(x) I)
        z_hat = netGz(x)
        mu, log_sigma = z_hat[:, :opt.nz], z_hat[:, opt.nz:]
        sigma = log_sigma.exp()

        z_hat = mu + sigma * epsilon
        x_hat = netGx(z)

        # train discriminator
        p_q = netDxz(torch.cat([netDx(x), netDz(z_hat)], 1))
        p_p = netDxz(torch.cat([netDx(x_hat), netDz(z)], 1))

        D_loss = - torch.mean(torch.log(p_q + eps)) - torch.mean(torch.log(1 - p_p + eps))
        D_loss.backward()  # Backpropagate loss
        optimizerD.step()  # Apply optimization step

        # init gradients
        # netDz.zero_grad(), netDx.zero_grad(), netDxz.zero_grad()
        netGx.zero_grad(), netGz.zero_grad()

        # train generator
        p_q = netDxz(torch.cat([netDx(x), netDz(z_hat.detach())], 1))
        p_p = netDxz(torch.cat([netDx(x_hat.detach()), netDz(z)], 1))

        G_loss = - torch.mean(torch.log(1 - p_q + eps)) - torch.mean(torch.log(p_p + eps))
        G_loss.backward()  # Backpropagate loss
        optimizerG.step()  # Apply optimization step

        ############################
        # Logging stuff
        ###########################

        print('[{}/{}][{}/{}] Loss_D: {} Loss_G: {}'
              .format(epoch, opt.epochs, batch_id, len(dataloader),
                      D_loss.data[0], G_loss.data[0]))

        # TODO(edgarriba): fixme since raises cuda out of memory
        # train_metrics.update(loss1=D_loss, loss2=G_loss, n=len(real_cpu))

    # Method 2 for logging: log Parent wrapper
    # (automatically logs all children)
    # xp.log_metric(train_metrics)


def test(dataloader, epoch):
    real_cpu_first, _ = iter(dataloader).next()

    if opt.cuda:
        real_cpu_first = real_cpu_first.cuda()

    netGx.eval(), netGz.eval()  # switch to test mode
    latent = netGz(Variable(real_cpu_first, volatile=True))

    # removes last sigmoid activation to visualize reconstruction correctly
    mu, sigma = latent[:, :opt.nz], latent[:, opt.nz:].exp()
    recon = nn.Sequential(*list(netGx.main.children())[:-1])(mu + sigma)

    vutils.save_image(recon.data, '{0}/reconstruction.png'.format(opt.experiment))
    vutils.save_image(real_cpu_first, '{0}/real_samples.png'.format(opt.experiment))


# MAIN LOOP

for epoch in range(opt.epochs):
    # reset training metrics
    train_metrics.reset()

    # call train/test routines
    train(dataloader, epoch)
    test(dataloader, epoch)

    # do checkpointing
    torch.save(netGx.state_dict(),
               '{0}/netGx_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netGz.state_dict(),
               '{0}/netGz_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netDx.state_dict(),
               '{0}/netDx_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netDz.state_dict(),
               '{0}/netDz_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netDxz.state_dict(),
               '{0}/netDxz_epoch_{1}.pth'.format(opt.experiment, epoch))