import torch
from torch import nn


def compl_mult(input,weights):
    return torch.einsum("bix,iox->box",input,weights)


class SpectralLayer(nn.Module):
    def __init__(self,in_channels,out_channels,modes):
        super(SpectralLayer,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.modes=modes
        self.scale=1/(in_channels*out_channels)
        self.weights=nn.Parameter(self.scale*torch.rand(in_channels,out_channels,self.modes,dtype=torch.cfloat))

    def forward(self,x):
        b=x.shape[0]
        n_spatial=x.size(-1)
        x_ft=torch.fft.rfft(x,dim=-1)
        n_freq=x_ft.size(-1)
        m=min(self.modes,n_freq)
        out_ft=torch.zeros(b,self.out_channels,n_freq,device=x.device,dtype=x_ft.dtype)
        out_ft[:,:,:m]=compl_mult(x_ft[:,:,:m],self.weights[:,:,:m])
        x=torch.fft.irfft(out_ft,n=n_spatial,dim=-1)
        return x


class FNO(nn.Module):
    def __init__(self,n_lifting,modes):
        super(FNO,self).__init__()
        self.n_lifting=n_lifting
        self.modes=modes
        self.spectral1=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.spectral2=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.spectral3=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.conv1=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.conv2=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.conv3=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.lifting=nn.Linear(2,self.n_lifting)
        self.activation=torch.nn.GELU()
        self.linear_post=nn.Linear(self.n_lifting,32)
        self.linear_final=nn.Linear(32,1)
    def FourierLayer(self,x,spect,conv):
        return self.activation(spect(x)+conv(x))
    def forward(self,x):
        x=self.lifting(x)
        x=self.activation(x)
        x=x.permute(0,2,1)
        x=self.FourierLayer(x,self.spectral1,self.conv1)
        x=self.FourierLayer(x,self.spectral2,self.conv2)
        x=self.FourierLayer(x,self.spectral3,self.conv3)
        x=x.permute(0,2,1)
        x=self.linear_post(x)
        x=self.activation(x)
        x=self.linear_final(x)
        x=x.squeeze(-1)
        return x


class FILMLayer(nn.Module):
    def __init__(self,channels):
        super(FILMLayer,self).__init__()
        self.scale=nn.Linear(1,channels,bias=True)
        self.bias=nn.Linear(1,channels,bias=True)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)
    def forward(self,x,time):
        time=time.reshape(-1,1).type_as(x)
        scale=self.scale(time).unsqueeze(2).expand_as(x)
        bias=self.bias(time).unsqueeze(2).expand_as(x)
        return x*(1.+scale)+bias


class tdFNO(nn.Module):
    def __init__(self,n_lifting,modes):
        super(tdFNO,self).__init__()
        self.n_lifting=n_lifting
        self.modes=modes
        self.spectral1=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.spectral2=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.spectral3=SpectralLayer(self.n_lifting,self.n_lifting,self.modes)
        self.conv1=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.conv2=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.conv3=nn.Conv1d(self.n_lifting,self.n_lifting,1)
        self.film1=FILMLayer(self.n_lifting)
        self.film2=FILMLayer(self.n_lifting)
        self.film3=FILMLayer(self.n_lifting)
        self.lifting=nn.Linear(3,self.n_lifting)
        self.activation=torch.nn.GELU()
        self.linear_post=nn.Linear(self.n_lifting,32)
        self.linear_final=nn.Linear(32,1)
    def FourierLayer(self,x,time,spect,conv,film):
        return self.activation(film(spect(x)+conv(x),time))
    def forward(self,x,time):
        x=self.lifting(x)
        x=self.activation(x)
        x=x.permute(0,2,1)
        x=self.FourierLayer(x,time,self.spectral1,self.conv1,self.film1)
        x=self.FourierLayer(x,time,self.spectral2,self.conv2,self.film2)
        x=self.FourierLayer(x,time,self.spectral3,self.conv3,self.film3)
        x=x.permute(0,2,1)
        x=self.linear_post(x)
        x=self.activation(x)
        x=self.linear_final(x)
        x=x.squeeze(-1)
        return x
