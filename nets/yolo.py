import torch
import torch.nn as nn
from nets.new_shufflenet import ShuffleFuse
from nets.modules_utils import SPPCSPC_tiny, Upsample, BaseConv, ELAN, DWConv, RepConv, fuse_conv_and_bn, \
    SpatialAttention, ChannelAttention


class YoloHead(nn.Module):
    def __init__(self, num_classes, in_channels=[64, 128, 256], act=True, depthwise=False, width=1.5):
        super(YoloHead, self).__init__()
        Conv = DWConv if depthwise else BaseConv

        self.rep_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            self.rep_convs.append(nn.Sequential(
                RepConv(in_channels[i], int(in_channels[i] * width), 3, 1, act=act),
                BaseConv(int(in_channels[i] * width), int(in_channels[0] * width), 1, 1, act=act)
            ))
            inp = int(in_channels[0] * width)  # 三个通道分别先升维到96,192,384,再统一
            # 降维到96
            self.cls_convs.append(nn.Sequential(
                Conv(inp, inp, 3, 1, act=act),
                Conv(inp, inp, 3, 1, act=act),
            ))
            self.cls_preds.append(
                nn.Conv2d(inp, num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_convs.append(nn.Sequential(
                Conv(inp, inp, 3, 1, act=act),
                Conv(inp, inp, 3, 1, act=act),
            ))
            self.reg_preds.append(
                nn.Conv2d(inp, out_channels=4, kernel_size=1, stride=1)
            )
            self.obj_preds.append(
                nn.Conv2d(inp, out_channels=1, kernel_size=1, stride=1)
            )

    def forward(self, inputs):
        outputs = []
        for i, x in enumerate(inputs):
            x = self.rep_convs[i](x)

            cls_feat = self.cls_convs[i](x)
            cls_out = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_out = self.reg_preds[i](reg_feat)

            obj_out = self.obj_preds[i](reg_feat)
            # 堆叠最终形成 bs,5+num_class,80,80
            output = torch.cat((reg_out, obj_out, cls_out), dim=1)
            outputs.append(output)
        return outputs  #


class YoloBody(nn.Module):
    def __init__(self, num_classes, model_size='Medium', depthwise=False):
        super(YoloBody, self).__init__()
        self.backbone = ShuffleFuse(model_size)
        if self.backbone.model_size == "Large":
            in_filters = [256, 512, 1024]
        else:
            in_filters = [128, 256, 512]
        # 20,20,512 -> 20,20,256
        self.spp = SPPCSPC_tiny(in_filters[2], in_filters[1], act=True)
        # 20,20,256 -> 40,40,128
        self.upsample1 = Upsample(in_filters[1], in_filters[0], act=True)
        # 40,40,256 -> 40,40,128
        self.conv_P4 = BaseConv(in_filters[1], in_filters[0], ksize=1, stride=1, act=True)

        # 40,40,256 -> 40,40,128
        self.elan1 = ELAN(in_filters[1], in_filters[0], act=True)
        # 40,40,128 -> 80,80,64
        self.upsample2 = Upsample(in_filters[0], int(in_filters[0] * 0.5), act=True)
        # 80,80,128 -> 80,80,64
        self.conv_P3 = BaseConv(in_filters[0], int(in_filters[0] * 0.5), ksize=1, stride=1, act=True)

        # attention1
        # self.ca1 = ChannelAttention(in_filters[0])
        # self.sa1 = SpatialAttention()

        # 80,80,128 -> 80,80,64
        self.elan2 = ELAN(in_filters[0], int(in_filters[0] * 0.5), act=True)
        # 80,80,64 -> 40,40,128
        self.conv_Pan3 = BaseConv(int(in_filters[0] * 0.5), in_filters[0], ksize=3, stride=2, act=True)

        # attention2
        # self.ca2 = ChannelAttention(in_filters[1])
        # self.sa2 = SpatialAttention()

        # 40,40,256 -> 40,40,128
        self.elan3 = ELAN(in_filters[1], in_filters[0], act=True)
        # 40,40,128 -> 20,20,256
        self.conv_Pan4 = BaseConv(in_filters[0], in_filters[1], ksize=3, stride=2, act=True)
        # 20,20,512 -> 20,20,256
        self.elan4 = ELAN(in_filters[2], in_filters[1], act=True)
        self.head = YoloHead(num_classes, [int(c * 0.5) for c in in_filters], act=True, depthwise=depthwise)

    def forward(self, x):
        # 经过第一个stage 160,160,64 -> 80,80,128 p3
        # 经过第二个stage 80,80,128 -> 40,40,256 p4
        # 经过第三个stage 40,40,256 -> 20,20,512 p5
        feat1, feat2, feat3 = self.backbone(x)
        p5 = self.spp(feat3)  # 20,20,512 -> 20,20,256

        p5_upsample = self.upsample1(p5)  # 20,20,256 -> 40,40,128
        p4 = self.elan1(torch.cat((self.conv_P4(feat2), p5_upsample), dim=1))  # 堆叠40,40,256 -> 40,40,128

        p4_upsample = self.upsample2(p4)
        p3 = torch.cat((self.conv_P3(feat1), p4_upsample), dim=1)
        # p3 = p3 * self.ca1(p3)
        # p3 = p3 * self.sa1(p3)

        pan3 = self.elan2(p3)  # 80,80,64

        pan4 = torch.cat((self.conv_Pan3(pan3), p4), dim=1)  # 加了注意力机制模块
        # pan4 = pan4 * self.ca2(pan4)
        # pan4 = pan4 * self.sa2(pan4)
        pan4 = self.elan3(pan4)  # 40,40,128

        pan5 = self.elan4(torch.cat((self.conv_Pan4(pan4), p5), dim=1))  # 20,20,256
        outputs = self.head.forward((pan3, pan4, pan5))
        return outputs  # 三个输出由浅导深

    def fuse(self):
        print('Fusing layers....')
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is BaseConv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 将卷积块和bn层进行fuse
                delattr(m, "bn")  # 删除bn层
                m.forward = m.fuseforward  # 在inference时forward函数改为fuseforward
        return self


if __name__ == '__main__':
    model = YoloBody(num_classes=1)
    x = torch.rand((1, 3, 640, 640))
    outs = model(x)
    for out in outs:
        print(out.shape)
