import torch
import torch.nn as nn
from nets.modules_utils import BaseConv, Shufflenet, ELAN, DWConv


# backbone网络
class ShuffleFuse(nn.Module):
    def __init__(self, model_size="Medium"):
        super(ShuffleFuse, self).__init__()
        self.model_size = model_size
        if model_size == 'Large':
            self.stage_out_channels = [-1, 32, 64, 128, 256, 512, 1024]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 32, 64, 128, 256, 512]
        else:
            raise NotImplementedError
        # 640,640,3 -> 320,320,32
        self.first_block = nn.Sequential(
            BaseConv(3, self.stage_out_channels[1], ksize=3, stride=2, groups=1, bias=False, act=True),  # 过渡
            DWConv(self.stage_out_channels[1], self.stage_out_channels[2], ksize=3, stride=1, act=True)
        )
        # 320,320,32 -> 160,160,64
        self.mp = nn.Sequential(
            BaseConv(self.stage_out_channels[2], self.stage_out_channels[2], ksize=3, stride=2, act=True),  # 32 -> 32
            DWConv(self.stage_out_channels[2], self.stage_out_channels[3], ksize=3, stride=1, act=True),  # 32 -> 64
            ELAN(self.stage_out_channels[3], self.stage_out_channels[3], act=True)  # 64 -> 64
        )

        in_channels = self.stage_out_channels[3]
        self.features = []
        self.stage_repeats = [4, 8, 4]  # repeat of stage
        # 经过第一个stage 160,160,64 -> 80,80,128
        # 经过第二个stage 80,80,128 -> 40,40,256
        # 经过第三个stage 40,40,256 -> 20,20,512
        for stage_id in range(len(self.stage_repeats)):
            nr = self.stage_repeats[stage_id]  # number of repeat
            out_channels = self.stage_out_channels[stage_id + 4]
            # act = nn.Hardswish() if stage_id > 0 else True  # use 'silu' in the first stage
            act = True
            # use_attention
            for i in range(nr):
                if i == 0:
                    self.features.append(
                        Shufflenet(in_channels, out_channels, base_mid_channels=out_channels // 2, ksize=3, stride=2,
                                   activation=act))
                else:
                    self.features.append(
                        Shufflenet(in_channels // 2, out_channels, base_mid_channels=out_channels // 2, ksize=3,
                                   stride=1, activation=act))
                in_channels = out_channels
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.mp(self.first_block(x))
        feat1 = self.features[0:4](x)
        feat2 = self.features[4:12](feat1)
        feat3 = self.features[12:16](feat2)
        return feat1, feat2, feat3


if __name__ == '__main__':
    model = ShuffleFuse()
    print(model)
