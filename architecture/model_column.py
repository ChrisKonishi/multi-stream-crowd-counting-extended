import torch.nn as nn
from architecture.network import Conv2d, Upsample

class Column_U(nn.Module):
    def __init__(self, kernel_size, first_out_chn, bn=False):
        super().__init__()
        self.entry = Conv2d(1, first_out_chn, kernel_size+2, same_padding=True, bn=bn)
        self.red_1 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(first_out_chn, 2*first_out_chn, kernel_size, same_padding=True, bn=bn)
        )
        self.red_2 = nn.Sequential(
            nn.MaxPool2d(2)
            , Conv2d(2*first_out_chn, first_out_chn, kernel_size, same_padding=True, bn=bn)
            , Conv2d(first_out_chn, int(first_out_chn/2), kernel_size, same_padding=True, bn=bn)
        )
        self.up_1 = Upsample(int(first_out_chn/2), int(first_out_chn/4), int(first_out_chn/4), int(first_out_chn/4), kernel_size=kernel_size, bn=bn)
        self.up_2 = Upsample(int(first_out_chn/4), int(first_out_chn/4), int(first_out_chn/4), int(first_out_chn/4), kernel_size=kernel_size, bn=bn)

        self.skip_1 = Conv2d(first_out_chn, int(first_out_chn/4), 1, same_padding=True, bn=bn)
        self.skip_2 = Conv2d(2*first_out_chn, int(first_out_chn/4), 1, same_padding=True, bn=bn)

    def forward(self, x):
        hold_1 = self.entry(x)
        hold_2 = self.red_1(hold_1)
        x = self.red_2(hold_2)

        x = self.up_1(x, self.skip_2(hold_2))
        x = self.up_2(x, self.skip_1(hold_1))

        return x
