import torch
import torch.nn as nn


# silu激活函数
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# h-swish激活函数
class HS(nn.Module):

    def __init__(self):
        super(HS, self).__init__()

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip


# 自动计算padding
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# 标准的conv + bn + act模块
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act=True):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):  # 在inference的时候进行融合bn和conv
        return self.act(self.conv(x))


# DWConv：包含一次dw卷积和pw卷积，也就是一次完整的深度可分离卷积
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act=True):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


# 最大池化操作
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


# shufflenet基本模块
class Shufflenet(nn.Module):

    def __init__(self, inp, oup, base_mid_channels, *, ksize, stride, activation=True, useSE=False):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup // 2  # 输出channel的一半

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp
        # 右分支
        # branch_main = [
        #     # pw
        #     nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(base_mid_channels),
        #     None,
        #     # dw
        #     nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
        #     nn.BatchNorm2d(base_mid_channels),
        #     # pw-linear
        #     nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(outputs),
        #     None,
        # ]
        branch_main = [
            BaseConv(inp, base_mid_channels, 1, 1, act=activation),
            BaseConv(base_mid_channels, base_mid_channels, ksize, stride, groups=base_mid_channels,
                     act=False),
            BaseConv(base_mid_channels, outputs, 1, 1, act=activation)
        ]

        # 判断使用哪个激活函数
        # if activation:
        #     assert useSE == False
        #     '''This model should not have SE with ReLU'''
        #     branch_main[2] = get_activation("silu")
        #     branch_main[-1] = get_activation("silu")
        # else:
        #     branch_main[2] = get_activation("hs")
        #     branch_main[-1] = get_activation("hs")
        #     if useSE:
        #         branch_main.append(SELayer(outputs))
        if useSE:
            branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            # branch_proj = [
            #     # dw
            #     nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
            #     nn.BatchNorm2d(inp),
            #     # pw-linear
            #     nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
            #     nn.BatchNorm2d(inp),
            #     None,
            # ]
            branch_proj = [
                BaseConv(inp, inp, ksize, stride, groups=inp, act=False),
                BaseConv(inp, inp, 1, 1, act=activation)
            ]
            # if activation == 'silu':
            #     branch_proj[-1] = get_activation("silu")
            # else:
            #     branch_proj[-1] = get_activation("hs")
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


# shuffle操作
def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


# SE 模块
class SELayer(nn.Module):

    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            # if the input is (N, C, H, W)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
            )
        else:
            # if the input is (N, C)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(inplanes, inplanes // 4, bias=False),
                nn.BatchNorm1d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // 4, inplanes, bias=False),
            )

    def forward(self, x):
        atten = self.SE_opr(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten


# spp模块
class SPPCSPC_tiny(nn.Module):
    def __init__(self, in_channel, out_channel, k=(5, 9, 13), act=True):
        super(SPPCSPC_tiny, self).__init__()
        hidden_channel = int(in_channel * 0.5)  # 256
        self.conv1 = BaseConv(in_channel, hidden_channel, ksize=1, stride=1, act=act)
        self.conv2 = BaseConv(in_channel, hidden_channel, ksize=1, stride=1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv3 = BaseConv(4 * hidden_channel, hidden_channel, ksize=1, stride=1, act=act)
        self.conv4 = BaseConv(2 * hidden_channel, out_channel, ksize=1, stride=1, act=act)

    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.conv3(torch.cat([x1] + [m(x1) for m in self.m], 1))
        y2 = self.conv2(x)
        return self.conv4(torch.cat((y1, y2), dim=1))


# 上采样模块，包含一次卷积和上采样
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, act=True):
        super(Upsample, self).__init__()
        self.conv1 = BaseConv(in_channel, out_channel, ksize=1, stride=1, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.upsample(self.conv1(x))


# ELAN模块
class ELAN(nn.Module):
    def __init__(self, in_channel, out_channel, act=True):
        super(ELAN, self).__init__()
        hidden_channel = int(out_channel * 0.5)
        self.conv1 = BaseConv(in_channel, hidden_channel, 1, 1, act=act)
        self.conv2 = BaseConv(in_channel, hidden_channel, 1, 1, act=act)
        self.conv3 = BaseConv(hidden_channel, hidden_channel, 3, 1, act=act)
        self.conv4 = BaseConv(hidden_channel, hidden_channel, 3, 1, act=act)
        self.con_fuse = BaseConv(hidden_channel * 4, out_channel, 1, 1, act=act)

    def forward(self, x):
        y1 = self.conv2(x)
        y2 = self.conv3(y1)
        y3 = self.conv4(y2)
        y4 = self.conv1(x)

        return self.con_fuse(torch.cat((y3, y2, y1, y4), dim=1))


# 融合conv和bn
def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv


class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# 注意力模块用法：
# out = self.ca(out) * out
# out = self.sa(out) * out


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 沿着通道求平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度求最大值
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    model = BaseConv(3, 32, 1, 1, act='')
    x = torch.rand((1, 3, 640, 640))
    outs = model(x)
