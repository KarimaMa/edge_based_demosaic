import torch
import torch.nn as nn
import numpy as np
from config import *
import ops
import task_config

class InterpSelector(nn.Module):
    def __init__(self, temp):
        self.temp = temp
        k = 5
        self.conv1 = torch.nn.Conv2d(1, 10, (k,k), bias=False, padding=int((k-1)/2))
        self.relu1 = torch.nn.ReLU()
        k = 3
        self.conv2 = torch.nn.Conv2d(10, 10, (k,k), bias=False, padding=int((k-1)/2))
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(10, 3, (1,1), bias=False, padding=int((k-1)/2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.softmax(x/self.temp)
        return x

class LaplacianWeightVert(nn.Module):
    def __init__(self, k):
        self.k = k
        self.g_interp = torch.nn.Conv2d(1, 1, (k, 1), bias=False, padding=int((k-1)/2), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (k, 1), bias=False, padding=int((k-1)/2), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb + 1e-8), torch.tensor(1.0))


class LaplacianWeightHoriz(nn.Module):
    def __init__(self, k):
        self.k = k
        self.g_interp = torch.nn.Conv2d(1, 1, (1, k), bias=False, padding=int((k-1)/2), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (1, k), bias=False, padding=int((k-1)/2), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb + 1e-8), torch.tensor(1.0))

class LaplacianWeight2D(nn.Module):
    def __init__(self, k):
        self.k = 5
        self.g_interp = torch.nn.Conv2d(1, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb + 1e-8), torch.tensor(1.0))


class GreenInterp(nn.Module):
    def __init__(self, k, temp):
        self.k = k
        self.temp = temp
        self.vertical = torch.nn.Conv2d(1, 1, (k, 1), bias=False, padding=int((k-1)/2), stride=1)
        self.chroma_vert = torch.nn.Conv2d(2, 1, (k, 1), bias=False, padding=int((k-1)/2), stride=1)
        self.lweight_vert = LaplacianWeightVert()

        self.horizontal = torch.nn.Conv2d(1, 1, (1, k), bias=False, padding=int((k-1)/2), stride=1)
        self.chorma_horiz = torch.nn.Conv2d(2, 1, (1, k), bias=False, padding=int((k-1)/2), stride=1)
        self.lweight_horiz = LaplacianWeightHoriz()

        self.2D = torch.nn.Conv2d(1, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.chorma_2D = torch.nn.Conv2d(2, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.lweight_2D = LaplacianWeight2D()

        self.InterpSelector = InterpSelector(temp)

    def forward(self, g, rb, bayer):
        selection = self.InterpSelector(bayer)
        g_vert = self.vertical(g)
        chroma_vert = self.chroma_vert(rb)
        lweight_vert = self.lweight_vert(g, rb)
        vert = g_vert + lweight_vert * chroma_vert 

        g_horiz = self.horizontal(g)
        chroma_horiz = self.chroma_horiz(rb)
        lweight_horiz = self.lweight_horiz(g, rb)
        horiz = g_horiz + lweight_horiz * chroma_horiz

        g_2D = self.2D(g)
        chroma_2D = self.chroma_2D(rb)
        lweight_2D = self.lweight_2D(g, rb)
        2D = g_2D + lweight_2D * chroma_2D

        g_interps = torch.stack((vert, horiz, 2D), dim=1)
        weighted_interps = interps * selection

        out = torch.sum(weighted_interps, dim=1)
        return out


class ChromaInterp(nn.Module):
    def __init__(self, k):
        self.k = k
        self.interp = torch.nn.Conv2d(1, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)

    def forward(self, x):
        return self.interp(x)


class Demosaic(nn.Module):
    def __init__(self, k, temp):
        self.temp = temp
        self.k = k
        self.green_interp = GreenInterp(k, temp)

        self.chroma_interp_h = ChromaInterp(k)
        self.chroma_interp_v = ChromaInterp(k)
        self.chroma_interp_q = ChromaInterp(k)

    def forward(self, bayer, r, g, b):
        rb = torch.stack((r, b), dim=1)
        
        green_pred = self.green_interp(g, rb, bayer)

        missing_g = green_pred * missing_gmask
        bayer_minus_ghat = bayer - missing_g

        chroma_interp_h = self.chroma_interp_h(bayer_minus_ghat)
        chroma_interp_v = self.chroma_interp_v(bayer_minus_ghat)
        chroma_interp_q = self.chroma_interp_q(bayer_minus_ghat)
        """
        chroma_h at gb is blue
        chroma_h at gr is red
        chroma_v at gb is red
        chroma_v at gr is blue
        chroma_q at b is red
        chroma_q at r is blue
        """
        missing_blue = chroma_interp_h * gb_mask +\
                chroma_interp_v * gr_mask +\
                chroma_interp_q * r_mask
        missing_red = chroma_interp_h * gr_mask +\
                chroma_interp_v * gb_mask +\
                chroma_interp_q * b_mask

        full_r = r + missing_r
        full_g = g + missing_g
        full_b = b + missing_b
        
        image = torch.stack((full_r, full_g, full_b), dim=1)
        return image



