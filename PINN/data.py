import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset


def sample_coefficients(K,seed):
    rng=np.random.RandomState(seed)
    a=rng.randn(K,K)
    return a

def generate_grid(N:int):
    v=[i/(N-1)for i in range(N)]
    grid=[(x,y)for x in v for y in v]
    return np.array(grid)

def index_matrix(K,r,seed):
    A=sample_coefficients(K,seed)
    i=np.arange(1,K+1)
    M=(i[:,None]**2+i[None,:]**2)**r
    return A*M

def f_dataset(N,K,r,seed):
    A=index_matrix(K,r,seed)
    v=np.linspace(0,1,N)
    k=np.arange(1,K+1)
    Sx=np.sin(np.pi*np.outer(v,k))
    Sy=Sx
    F=(np.pi/(K**2))*(Sx@A@Sy.T)
    return F.reshape(-1)

def u_dataset(N,K,r,seed):
    A=index_matrix(K,r-1,seed)
    v=np.linspace(0,1,N)
    k=np.arange(1,K+1)
    Sx=np.sin(np.pi*np.outer(v,k))
    Sy=Sx
    U=(1.0/(np.pi*(K**2)))*(Sx@A@Sy.T)
    return U.reshape(-1)



def create_dataloaders(data,seed=0,batch_size=256):
    pos=data['pos']
    f=data['f']
    u=data['u']
    mask=(pos[:,0]>0)&(pos[:,0]<1)&(pos[:,1]>0)&(pos[:,1]<1)
    rng=np.random.default_rng(seed)
    splits=[]
    for indices in[np.where(mask)[0],np.where(~mask)[0]]:
        indices=rng.permutation(indices)
        split=int(0.8*len(indices))
        if split==0 or split==len(indices):
            raise ValueError('The grid needs enough interior and boundary points for a split.')
        splits.append((indices[:split],indices[split:]))
    int_train,int_test=splits[0]
    bd_train,bd_test=splits[1]
    train=np.concatenate([int_train,bd_train])
    test=np.concatenate([int_test,bd_test])
    pos=torch.as_tensor(pos,dtype=torch.float32)
    f=torch.as_tensor(f,dtype=torch.float32)
    u=torch.as_tensor(u,dtype=torch.float32)
    u_mean=u[train].mean()
    u_std=u[train].std().clamp_min(1e-12)
    interior=TensorDataset(pos[int_train],f[int_train],u[int_train])
    boundary=TensorDataset(pos[bd_train],u[bd_train])
    dd=TensorDataset(pos[train],(u[train]-u_mean)/u_std)
    generator=torch.Generator().manual_seed(seed)
    return{
        'training_interior':DataLoader(interior,batch_size=batch_size,shuffle=True,generator=generator),
        'training_boundary':DataLoader(boundary,batch_size=max(1,round(batch_size*len(bd_train)/len(int_train))),shuffle=True,generator=generator),
        'train_loader':DataLoader(dd,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)),
        'test_pos':pos[test],'test_u':u[test],
        'test_int_pos':pos[int_test],'test_int_u':u[int_test],
        'test_interior':(pos[int_test],f[int_test],u[int_test]),
        'test_boundary':(pos[bd_test],u[bd_test]),
        'u_mean':u_mean,'u_std':u_std
    }


def generate_data(N,K,r=0.5,seed=0):
    if N<4 or K<1 or K>=N-1:
        raise ValueError('Use N>=4 and 1<=K<N-1 to resolve the sine modes.')
    return{'pos':generate_grid(N),'f':f_dataset(N,K,r,seed),'u':u_dataset(N,K,r,seed)}
