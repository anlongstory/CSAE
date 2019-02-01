from Basic_blocks import *

class encoder_C(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(encoder_C,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True))

        self.block2 = nn.Sequential(
            BasicBlock(16, 32, False),
            BasicBlock(32, 64, False),
            BasicBlock(64, 64, False),
        )

        self.Linear_down = nn.Linear(64 * 4 * 4, out_channels)


    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x = x.view(x.shape[0], -1)
        out = self.Linear_down(x)
        return out

class decoder_C(nn.Module):
    def __init__(self,in_channels):
        super(decoder_C,self).__init__()
        self.Linear_up = nn.Linear(in_channels, 64*4*4)

        self.deconvBlock1 = nn.Sequential(
            DecodeBlock(64, 64)
        )
        self.deconvBlock2 = nn.Sequential(
            DecodeBlock(64, 32)
        )
        self.deconvBlock3 = nn.Sequential(
            DecodeBlock(32, 16)
        )
        self.conv1 = nn.Conv2d(16, 1, 1)

    def forward(self,x):
        x = self.Linear_up(x)
        x = x.view(-1,64,4,4)
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        x = self.deconvBlock3(x)
        x = self.conv1(x)
        return x

class encoder_L(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(encoder_L,self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(in_channels,400),
            nn.ReLU(True)
        )

        self.Linear_down = nn.Linear(400, out_channels)

    def forward(self,x):
        x=x.view(-1,28*28)
        x=self.block1(x)
        out = self.Linear_down(x)
        return out

class decoder_L(nn.Module):
    def __init__(self,in_channels):
        super(decoder_L,self).__init__()

        self.deconvBlock1 = nn.Sequential(
            nn.Linear(in_channels, 400),
            nn.ReLU()
        )

        self.deconvBlock2 = nn.Sequential(
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        return x

