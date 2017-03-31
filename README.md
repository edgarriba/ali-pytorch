# Adversarially Learned Inference
Implementation of paper [Aversarially Learned Inference](https://arxiv.org/abs/1606.00704) in Pytorch

`main.py` includes training code for datasets
- [X] SVHN
- [ ] CIFAR10
- [ ] CelebA

`models.py` includes the network architectures for the different datasets as defined in the orginal paper

## Usage
```
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batch-size BATCH_SIZE] [--image-size IMAGE_SIZE] [--nc NC]
               [--nz NZ] [--epochs EPOCHS] [--lr LR] [--beta1 BETA1]
               [--beta2 BETA2] [--cuda] [--ngpu NGPU] [--gpu-id GPU_ID]
               [--netGx NETGX] [--netGz NETGZ] [--netDz NETDZ] [--netDx NETDX]
               [--netDxz NETDXZ] [--clamp_lower CLAMP_LOWER]
               [--clamp_upper CLAMP_UPPER] [--experiment EXPERIMENT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | svhn | celeba
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batch-size BATCH_SIZE
                        input batch size
  --image-size IMAGE_SIZE
                        the height / width of the input image to network
  --nc NC               input image channels
  --nz NZ               size of the latent z vector
  --epochs EPOCHS       number of epochs to train for
  --lr LR               learning rate for optimizer, default=0.00005
  --beta1 BETA1         beta1 for adam. default=0.5
  --beta2 BETA2         beta2 for adam. default=0.999
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --gpu-id GPU_ID       id(s) for CUDA_VISIBLE_DEVICES
  --netGx NETGX         path to netGx (to continue training)
  --netGz NETGZ         path to netGz (to continue training)
  --netDz NETDZ         path to netDz (to continue training)
  --netDx NETDX         path to netDx (to continue training)
  --netDxz NETDXZ       path to netDxz (to continue training)
  --clamp_lower CLAMP_LOWER
  --clamp_upper CLAMP_UPPER
  --experiment EXPERIMENT
                        Where to store samples and models
```
## Example
command line example for training SVHN
```
python main.py --dataset svhn --dataroot . --experiment svhn_ali --cuda --ngpu 1 --gpu-id 1 --batch-size 100 --epochs 100 --image-size 32 --nz 256 --lr 1e-4 --beta1 0.5 --beta2 10e-3
```

## Cite
```
@article{DBLP:journals/corr/DumoulinBPLAMC16,
  author    = {Vincent Dumoulin and
               Ishmael Belghazi and
               Ben Poole and
               Alex Lamb and
               Mart{\'{\i}}n Arjovsky and
               Olivier Mastropietro and
               Aaron C. Courville},
  title     = {Adversarially Learned Inference},
  journal   = {CoRR},
  volume    = {abs/1606.00704},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.00704},
}
```
