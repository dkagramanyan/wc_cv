import pickle

import torch
from torch import distributed as dist
from torch.utils import data

from torch.nn import functional as F

from torch import nn
# from torch.utils.data import DataLoader
#
# from torchvision import datasets, transforms, utils

import os
import numpy as np

from tqdm.notebook import trange, tqdm
from PIL import Image
from matplotlib import pyplot as plt

from torchvision import datasets, transforms, utils
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import ImageFolder
from numpy import random

from skimage.filters import threshold_otsu, median

import time
from abc import abstractmethod

LOCAL_PROCESS_GROUP = None


class TripletFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        super(TripletFolder, self).__init__(root=root, transform=transform)

        # Create a dictionary of lists for each class for reverse lookup
        # to generate triplets 
        self.classdict = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdict[ci] = []

        # append each file in the approach dictionary element list
        for s in self.samples:
            self.classdict[s[1]].append(s[0])

        # keep track of the sizes for random sampling
        self.classdictsize = {}
        for c in self.classes:
            ci = self.class_to_idx[c]
            self.classdictsize[ci] = len(self.classdict[ci])

    # Return a triplet, with positive and negative selected at random.
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, sample, sample) where the samples are anchor, positive, and negative.
            The positive and negative instances are sampled randomly. 
        """

        # The anchor is the image at this index.
        a_path, a_target = self.samples[index]

        prs = random.random() # positive random sample
        nrc = random.random() # negative random class
        nrs = random.random() # negative random sample

        # random negative class cannot be the same class as anchor. We add
        # a random offset that is 1 less than the number required to wrap
        # back around to a_target after modulus. 
        nrc = (a_target + int(nrc*(len(self.classes) - 1))) % len(self.classes)

        # Positive Instance: select a random instance from the same class as anchor.
        p_path = self.classdict[a_target][int(self.classdictsize[a_target]*prs)]
        
        # Negative Instance: select a random instance from the random negative class.
        n_path = self.classdict[nrc][int(self.classdictsize[nrc]*nrs)]

        # Load the data for these samples.
        a_sample = self.loader(a_path)
        p_sample = self.loader(p_path)
        n_sample = self.loader(n_path)

        # apply transforms
        if self.transform is not None:
            a_sample = self.transform(a_sample)
            p_sample = self.transform(p_sample)
            n_sample = self.transform(n_sample)

        # note that we do not return the label! 
        return a_sample, p_sample, n_sample 



def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    if LOCAL_PROCESS_GROUP is None:
        raise ValueError("tensorfn.distributed.LOCAL_PROCESS_GROUP is None")

    return dist.get_rank(group=LOCAL_PROCESS_GROUP)


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=op)

    return tensor


def all_gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([1]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), 0)

    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = []
        values = []

        for k in sorted(input_dict.keys()):
            keys.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, 0)
        dist.reduce(values, dst=0)

        if dist.get_rank() == 0 and average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Trainer():

    @classmethod
    def train(cls, model, optimizer, train_loader, test_loader, model_path, epochs=100, device='cuda',
              latent_loss_weight=0.25, sample_size=25, print_text=False):

        if os.path.exists(model_path) is False:
            os.mkdir(model_path)

        for epoch in range(epochs):
            
            if is_primary():
                train_loader = tqdm(train_loader)

            criterion = nn.MSELoss()

            mse_sum = 0
            mse_n = 0
            test_mean_loss = []
            train_mean_loss = []

            for i, (img, label) in enumerate(train_loader):
                
                start=time.time()
                model.zero_grad()

                img = img.to(device)

                out, latent_loss = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()
                loss = recon_loss + latent_loss_weight * latent_loss
                loss.backward()
                train_mean_loss.append(loss.item())
                # if scheduler is not None:
                #     scheduler.step()
                optimizer.step()

                part_mse_sum = recon_loss.item() * img.shape[0]
                part_mse_n = img.shape[0]
                comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
                comm = all_gather(comm)

                for part in comm:
                    mse_sum += part["mse_sum"]
                    mse_n += part["mse_n"]
                
                end=time.time()
                if is_primary():
                    lr = optimizer.param_groups[0]["lr"]
                    text= f"epoch: {epoch + 1}; step: {i+1}/{len(train_loader)}; eta: {(end-start)*(len(train_loader)-i)/60:.1f} min; loss: {str(round(np.mean(train_mean_loss), 5))}; mse:{recon_loss.item():.5f}; latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; lr: {lr:.5f}"
                    train_loader.set_description(
                        (
                           text
                        )
                    )
                if print_text:
                    print(text)

                model.train()

            model.eval()

            with torch.no_grad():

                for j, (img, label) in enumerate(test_loader):
                    img = img.to(device)
                    out, latent_loss = model(img)
                    test_recon_loss = criterion(out, img)
                    test_latent_loss = latent_loss.mean()
                    test_loss = test_recon_loss + latent_loss_weight * latent_loss
                    test_mean_loss.append(round(test_loss.item(), 5))

                sample = img[:sample_size]

            utils.save_image(
                torch.cat([sample, out], 0),
                f"{model_path}/{str(epoch + 1).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                # range=(-1, 1),
            )

            print(f'test elbo: {str(round(np.mean(test_mean_loss), 5))}')
            torch.save(model.state_dict(),
                       f"{model_path}/vqvae_{str(epoch + 1).zfill(3)}_train_{str(round(np.mean(train_mean_loss), 5))}_test_{str(round(np.mean(test_mean_loss), 5))}.pt")


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=5,
            n_res_channel=32,
            embed_dim=8,
            n_embed=512,
            decay=0.99,
            umap_path=None,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.embed_dim=embed_dim
        if umap_path is not None:
            self.umap=umaped_vct_tb_loaded = pickle.load((open(umap_path, 'rb')))
            

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def encode_t(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        
        return quant_t

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec
    
    def decode_li(self, z,t_size=32,b_size=64, N=1024 ):
        
        # print(z.shape)
        quant_t, quant_b=z[:,:N], z[:, N:]
        
        quant_t=quant_t.reshape((-1, self.embed_dim, t_size, t_size))
        quant_b=quant_b.reshape((-1, self.embed_dim, b_size, b_size))
        
        print(quant_t.shape)
        print(quant_b.shape)
        
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec
    
    def decode_li_umap(self, z,t_size=32,b_size=64, N=1024 ):
        
        quant_tb=self.umap.inverse_transform(z)
        
        quant_t, quant_b=quant_tb[:,:N], quant_tb[:, N:]
        
        quant_t=quant_t.reshape((-1, self.embed_dim, t_size, t_size))
        quant_b=quant_b.reshape((-1, self.embed_dim, b_size, b_size))
        
        # print(quant_t.shape)
        # print(quant_b.shape)
        
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


class QuantizeAdaptive(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, n_embedded_l=None, dim_l=None):
        
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed[:,:n_embedded_l]
                + self.embed[:,:n_embedded_l].pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            all_reduce(embed_onehot_sum)
            all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        
        if dim_l!=None:
            quantize[:,:,:,dim_l:]=0
        
        # return quantize[:,:,:,:dim_l], diff, embed_ind
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
class Vqvae2Adaptive(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=5,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = QuantizeAdaptive(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = QuantizeAdaptive(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input,n_embedded_l=None, dim_l=None):
        quant_t, quant_b, diff, _, _ = self.encode(input,n_embedded_l=n_embedded_l, dim_l=dim_l)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input,n_embedded_l, dim_l):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t, n_embedded_l=n_embedded_l, dim_l=dim_l )
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b, n_embedded_l=n_embedded_l, dim_l=dim_l )

        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def encode_t_b(self, input, n_embedded_l=1, dim_l=1 ):
        enc_b = self.enc_b(input)
        
        enc_t = self.enc_t(enc_b)
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, _, _ = self.quantize_t(quant_t, n_embedded_l=n_embedded_l, dim_l=dim_l )
        quant_t = quant_t.permute(0, 3, 1, 2)
        
        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, _, _ = self.quantize_b(quant_b, n_embedded_l=n_embedded_l, dim_l=dim_l )

        quant_b = quant_b.permute(0, 3, 1, 2)

        return quant_t, quant_b

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels: int,
                 latent_dims: int,
                 hidden_dims: List[int] = None,
                 flow_check=False,
                 out_channels=3,
                 flag_128=False,
                 **kwargs) -> None:
        """Instantiates the VAE model

        Params:
            in_channels (int): Number of input channels
            latent_dims (int): Size of latent dimensions
            hidden_dims (List[int]): List of hidden dimensions
        """
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dims
        self.flow_check = flow_check

        if self.flow_check:
            self.flow = Flow(self.latent_dim, 'planar', 16)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dims)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()
            
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the convolutional network
        and outputs the latent variables.

        Params:
            input (Tensor): Input tensor [N x C x H x W]

        Returns:
            mu (Tensor) and log_var (Tensor) of latent variables
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        if self.flow_check:
            z, log_det = self.reparameterize(mu, log_var)
            return mu, log_var, z, log_det

        else:
            z = self.reparameterize(mu, log_var)
            return mu, log_var, z

    def decode(self, z):
        """
        Maps the given latent variables
        onto the image space.

        Params:
            z (Tensor): Latent variable [B x D]

        Returns:
            result (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1)

        Params:
            mu (Tensor): Mean of Gaussian latent variables [B x D]
            logvar (Tensor): log-Variance of Gaussian latent variables [B x D]

        Returns: 
            z (Tensor) [B x D]
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        if self.flow_check:
            return self.flow(z)

        else:
            return z

    def forward(self, input, **kwargs):

        if self.flow_check:
            mu, log_var, z, log_det = self.encode(input)

            return self.decode(z), mu, log_var, log_det

        else:
             mu, log_var, z = self.encode(input)

             return self.decode(z), mu, log_var

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.

        Params:
            num_samples (Int): Number of samples
            current_device (Int): Device to run the model

        Returns:
            samples (Tensor)
        """

        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x , **kwargs):
        """
        Given an input image x, returns the reconstructed image

        Params:
            x (Tensor): input image Tensor [B x C x H x W]

        Returns:
            (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class Vqvae2AdaptiveVae(Vqvae2Adaptive):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=5,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
        latent_dims=200, 
        in_channels=1, 
        out_channels=1,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = QuantizeAdaptive(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = QuantizeAdaptive(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.vae_top=VanillaVAE(in_channels=in_channels,latent_dims=latent_dims, out_channels=out_channels, hidden_dims=[32, 64, 128, 256, 512])
        self.vae_bottom=VanillaVAE(in_channels=in_channels,latent_dims=latent_dims, out_channels=out_channels, hidden_dims=[16, 32, 64, 128, 256, 512])
        

    def forward(self, input,n_embedded_l=None, dim_l=None):
        quant_t, quant_b, diff, _, _ = self.encode(input,n_embedded_l=n_embedded_l, dim_l=dim_l)
        quant_t_out, mu_t, log_var_t=self.vae_top(quant_t)
        quant_b_out, mu_b, log_var_b=self.vae_bottom(quant_b)
        dec = self.decode(quant_t_out, quant_b_out)

        elbo_t=self.elbo_loss(quant_t_out, quant_t, mu_t, log_var_t)
        elbo_b=self.elbo_loss(quant_b_out, quant_b, mu_b, log_var_b)

        return dec, diff, elbo_t, elbo_b

    def elbo_loss(self, recon_x, x, mu, logvar, beta=1):
        """
        ELBO Optimization objective for gaussian posterior
        (reconstruction term + regularization term)
        """
        reconstruction_function = nn.MSELoss(reduction='sum')
        MSE = reconstruction_function(recon_x, x)
    
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
    
        return MSE + beta*KLD





class Embeddings():

    @classmethod
    def get_vqvae2_embs(cls, vqvae2_model, dataset, device='cuda'):
        images_embs_t = []
        images_embs_b = []

        # vqvae2_model=model

        dataset_path = dataset.__dict__['root']
        classes_folders = os.listdir(dataset_path)
        classes_folders_images = [os.listdir(dataset_path + '/' + folder) for folder in classes_folders]
        classes_folders_images_num = [len(os.listdir(dataset_path + '/' + folder)) for folder in classes_folders]

        img_transform = dataset.__dict__['transform']

        for i in range(len(classes_folders)):
            print(f'Number of folders {i + 1}/{len(classes_folders)}')
            for j in tqdm(range(classes_folders_images_num[i]), desc=f'Folder {classes_folders[i]}'):
                image_path = dataset_path + '/' + classes_folders[i] + '/' + classes_folders_images[i][j]

                image = Image.open(image_path)
                image = image.convert("RGB")
                image = img_transform(image)
                image = image.unsqueeze(0).to(device)

                vqvae2_model.zero_grad()

                quant_t, quant_b, diff, _, indx_b = vqvae2_model.encode(image)

                swapped_t = quant_t.cpu().detach().numpy().flatten()
                swapped_b = quant_b.cpu().detach().numpy().flatten()

                images_embs_t.append(swapped_t)
                images_embs_b.append(swapped_b)

        return np.array(images_embs_t), np.array(images_embs_b)

    @classmethod
    def get_vqvae2_embs_adaptive(cls, vqvae2_model, dataset,n_embedded_l=-1,dim_l=-1,  device='cuda'):
        images_embs_t = []
        images_embs_b = []

        # vqvae2_model=model

        dataset_path = dataset.__dict__['root']
        classes_folders = os.listdir(dataset_path)
        classes_folders_images = [os.listdir(dataset_path + '/' + folder) for folder in classes_folders]
        classes_folders_images_num = [len(os.listdir(dataset_path + '/' + folder)) for folder in classes_folders]

        img_transform = dataset.__dict__['transform']

        for i in range(len(classes_folders)):
            print(f'Number of folders {i + 1}/{len(classes_folders)}')
            for j in tqdm(range(classes_folders_images_num[i]), desc=f'Folder {classes_folders[i]}'):
                image_path = dataset_path + '/' + classes_folders[i] + '/' + classes_folders_images[i][j]

                image = Image.open(image_path)
                image = image.convert("RGB")
                image = img_transform(image)
                image = image.unsqueeze(0).to(device)

                vqvae2_model.zero_grad()

                quant_t, quant_b= vqvae2_model.encode_t_b(image,n_embedded_l=n_embedded_l, dim_l=dim_l)

                swapped_t = quant_t.cpu().detach().numpy().flatten()
                swapped_b = quant_b.cpu().detach().numpy().flatten()

                images_embs_t.append(swapped_t)
                images_embs_b.append(swapped_b)

        return np.array(images_embs_t), np.array(images_embs_b)


    @classmethod
    def plot_2d_scatter_embs(cls, embs_scatter, legend, r_shape=(5, 360, 2), dot_size=20, fontsize=15, save=False,
                             name=None, N=15, M=15):
        """
        :param name:
        :param r_shape:
        :param embs_scatter:  ndarray shape(N_classes,N_images,2)
        :param legend:
        :param dot_size:
        :param fontsize:
        :param save:
        :param N:
        :param M:
        :return:
        """

        fig, ax = plt.subplots(figsize=(N, M))

        colors = ['b', 'g', 'y', 'm', 'c']
        markers = ['8', 'v', 's', 'd', '*', ]

        scatter_xy = embs_scatter.reshape(r_shape)

        for i in range(len(scatter_xy)):
            ax.scatter(scatter_xy[i, :, 0], scatter_xy[i, :, 1], color=colors[i], s=dot_size,
                       marker=markers[i])
            

        ax.legend(legend, fontsize=fontsize)
        if save and name:
            plt.savefig(f'{name}.png', bbox_inches = 'tight')
        plt.show()

        

class Trans(torch.nn.Module):

    def __init__(self, kernel_size=3):
        self.kernel_size=kernel_size
    
    def __call__(self, tensor):
        x=tensor.cpu().detach().numpy()

        x_temp=[]
        for x_i in x:
            x_temp.append(median(x_i, self.kernel_size))

        x=x_temp.copy()
        x_temp=[]
        for x_i in x:
            thresh = threshold_otsu(x_i)
            binary = x_i > thresh
            x_temp.append(x_i*255)

        x=x_temp.copy()
        return torch.FloatTensor(x) 
    
    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self.ketnel_size})"
