import torch


def boundary_loss(model,bd_batch):
    pos,u=bd_batch
    u=u.view(-1,1)
    output=model(pos)
    se=(output-u)**2
    return torch.mean(se)

def interior_loss(model,int_batch):
    pos,f,u=int_batch
    f=f.view(-1,1)
    u=u.view(-1,1)
    pos=pos.detach().requires_grad_(True)
    output=model(pos)
    g=torch.autograd.grad(output,pos,grad_outputs=torch.ones_like(output),create_graph=True)[0]
    ux=g[:,0:1]
    uy=g[:,1:2]
    uxx=torch.autograd.grad(ux,pos,grad_outputs=torch.ones_like(ux),create_graph=True)[0][:,0:1]
    uyy=torch.autograd.grad(uy,pos,grad_outputs=torch.ones_like(uy),create_graph=True)[0][:,1:2]
    lap=uxx+uyy
    r=-lap-f
    return torch.mean(r**2)

def total_pde_loss(model,int_batch,bd_batch,lambdabd):
    return interior_loss(model,int_batch)+lambdabd*boundary_loss(model,bd_batch)

def data_driven_loss(model,batch):
    pos,u=batch
    u=u.view(-1,1)
    output=model(pos)
    return torch.mean((output-u)**2)
