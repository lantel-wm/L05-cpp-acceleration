import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.CircularPadding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.CircularPadding(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
    
    def CircularPadding(self, inp):
        _, _, L = inp.shape
        kl = self.kernel_size
        sl = self.stride
        assert kl%2 != 0 and (L-kl)%sl==0, 'kernel_size should be odd, (dim-kernel_size) should be divisible by stride'

        pl = int((L - 1 - (L - kl) / sl) // 2)
        
        x = F.pad(inp, (pl, pl), 'circular')

        return x
        


class Down(nn.Module):
    def __init__(self, kernel_size, stride, in_channels) -> None:
        super().__init__()
        self.downblock = nn.Sequential(
            DoubleConv(in_channels, in_channels * 2, kernel_size, stride),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x = self.downblock(x)
        return x
    
    
class Up(nn.Module):
    def __init__(self, kernel_size, stride, in_channels) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, in_channels // 2, kernel_size, stride)
        
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
    
    def CircularPadding(self, inp):
        _, _, L = inp.shape
        kl = self.kernel_size
        sl = self.stride
        assert kl%2 != 0 and (L-kl)%sl==0, 'kernel_size should be odd, (dim-kernel_size) should be divisible by stride'

        pl = int((L - 1 - (L - kl) / sl) // 2)
        
        x = F.pad(inp, (pl, pl), 'circular')

        return x
    

class Unet(nn.Module):
    def __init__(self, c, c_expand=32, kernel_sizes=[], strides=[]):
        super().__init__()
        
        self.inconv = nn.Conv1d(in_channels=c, out_channels=c_expand, kernel_size=3, padding=0, stride=1)
        self.bn1 = nn.BatchNorm1d(c_expand)
        self.relu = nn.ReLU(inplace=True)
        
        self.down1 = Down(kernel_sizes[0], strides[0], c_expand)
        self.down2 = Down(kernel_sizes[1], strides[1], c_expand * 2)
        self.down3 = Down(kernel_sizes[2], strides[2], c_expand * 4)
        self.down4 = Down(kernel_sizes[3], strides[3], c_expand * 8)
        self.down5 = Down(kernel_sizes[4], strides[4], c_expand * 16)
        
        self.up1 = Up(kernel_sizes[4], strides[4], c_expand * 32)
        self.up2 = Up(kernel_sizes[3], strides[3], c_expand * 16)
        self.up3 = Up(kernel_sizes[2], strides[2], c_expand * 8)
        self.up4 = Up(kernel_sizes[1], strides[1], c_expand * 4)
        self.up5 = Up(kernel_sizes[0], strides[0], c_expand * 2)
        
        self.outconv = nn.Conv1d(in_channels=c_expand, out_channels=c, kernel_size=3, padding=0, stride=1)
        

    def forward(self, x):
        # in conv
        x = self.CircularPadding(x, 3, 1)
        x = self.inconv(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        x = self.CircularPadding(x, 3, 1)
        x = self.outconv(x)
        
        return x
    
    def CircularPadding(self, inp, kernel_size, stride):
        _, _, L = inp.shape
        kl = kernel_size
        sl = stride
        assert kl%2 != 0 and (L-kl)%sl==0, 'kernel_size should be odd, (dim-kernel_size) should be divisible by stride'

        pl = int((L - 1 - (L - kl) / sl) // 2)
        
        x = F.pad(inp, (pl, pl), 'circular')

        return x
    
if __name__ == '__main__':
    model = Unet(c=1, c_expand=32, kernel_sizes=[3, 3, 3, 3, 3], strides=[1, 1, 1, 1, 1])
    x = torch.rand((1, 1, 960))
    x = model(x)
    print(x.shape)