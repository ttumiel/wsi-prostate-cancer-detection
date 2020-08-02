import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import timm, math


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(p={:.4f}'.format(self.p.data.tolist()[0]) + ', eps=' + str(self.eps) + ')'

class EffNetPatch(nn.Module):
    def __init__(self, n, n_patches=9):
        super().__init__()
        self.n_patches = n_patches
        self.m=timm.create_model('efficientnet_b0', pretrained=True)
        nc = self.m.classifier.in_features
        self.m.global_pool = nn.Identity()
        self.m.classifier = nn.Identity()
#         self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(nc*2,n))
        self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Linear(nc,n))
        nn.init.kaiming_normal_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, x):
        bs, *s = x.shape
        x = x.view(bs*self.n_patches, *s[1:])
        x = self.m.forward_features(x)
        s = x.shape
        x = x.view(bs, self.n_patches, *s[1:]).permute(0,2,1,3,4).contiguous().view(bs,s[1], self.n_patches*s[2], s[3])
        x = self.head(x)
        return x

class EffNetConcat(nn.Module):
    def __init__(self, n, n_patches=9):
        super().__init__()
        self.n_patches = n_patches
        self.m=timm.create_model('efficientnet_b0', pretrained=True)
        nc = self.m.classifier.in_features
        self.m.classifier = nn.Linear(nc, n) # Identity()
#         self.head = nn.Sequential(AdaptiveConcatPool2d(),nn.Flatten(),nn.Linear(nc*2,n))
#         self.head = nn.Sequential(GeM(),nn.Flatten(),nn.Linear(nc,n))
        nn.init.kaiming_normal_(self.m.classifier.weight)
        nn.init.zeros_(self.m.classifier.bias)

    def forward(self, x):
        return self.m(x)
