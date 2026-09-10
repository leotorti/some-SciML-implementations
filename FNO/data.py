from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader


TIMES=(0.0,0.25,0.5,0.75,1.0)
DATA_DIR=Path(__file__).resolve().parent/'data'


def load_data(filename,data_dir=DATA_DIR,limit=None):
    path=Path(data_dir)/filename
    data=np.load(path,allow_pickle=False)
    if data.ndim!=3 or data.shape[1]!=len(TIMES)or data.shape[2]<2 or len(data)==0:
        raise ValueError(f'{path.name} must have shape (samples,5,space).')
    if not np.isfinite(data).all():
        raise ValueError(f'{path.name} contains nonfinite values.')
    if limit is not None:
        if limit<1:
            raise ValueError('limit must be positive.')
        data=data[:limit]
    return torch.as_tensor(data,dtype=torch.float32)


class EvolutionDataset(Dataset):
    def __init__(self,data,all_pairs=False):
        self.data=data
        self.x=torch.linspace(0,1,data.shape[-1])
        self.pairs=[(i,j)for i in range(5)for j in range(i+1,5)]if all_pairs else[(0,4)]

    def __len__(self):
        return len(self.data)*len(self.pairs)

    def __getitem__(self,index):
        sample,pair=divmod(index,len(self.pairs))
        i,j=self.pairs[pair]
        return self.x,self.data[sample,i],torch.tensor(TIMES[j]-TIMES[i]),self.data[sample,j]


def create_loader(data,all_pairs=False,batch_size=32,shuffle=False,seed=0):
    return DataLoader(EvolutionDataset(data,all_pairs),batch_size=batch_size,shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def model_input(x,u,time,time_dependent):
    features=[x,u]
    if time_dependent:
        features.append(time[:,None].expand_as(u))
    return torch.stack(features,dim=-1)
