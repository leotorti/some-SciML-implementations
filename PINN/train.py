import torch
from torch.nn.utils import parameters_to_vector
from.losses import interior_loss,boundary_loss,data_driven_loss


def snapshot(model):
    return parameters_to_vector(model.parameters()).detach().cpu().clone()


def fit(model,data,n_adam=500,n_lbfgs=500,lambdabd=400,kind='pinn',snapshot_every=10):
    if kind not in['pinn','dd']:
        raise ValueError('kind must be pinn or dd.')
    if n_adam<0 or n_lbfgs<0 or n_adam+n_lbfgs==0 or snapshot_every<1:
        raise ValueError('Choose a positive training length and snapshot interval.')
    model.train()
    history=[]
    trajectory=[snapshot(model)]
    int_batch=data['training_interior'].dataset.tensors
    bd_batch=data['training_boundary'].dataset.tensors
    dd_batch=data['train_loader'].dataset.tensors

    def full_loss():
        if kind=='pinn':
            loss_int=interior_loss(model,int_batch)
            loss_bd=boundary_loss(model,bd_batch)
            return loss_int+lambdabd*loss_bd,loss_int,loss_bd
        loss=data_driven_loss(model,dd_batch)
        return loss,loss.new_zeros(()),loss.new_zeros(())

    def record(epoch,phase):
        loss,loss_int,loss_bd=full_loss()
        history.append({'epoch':epoch,'phase':phase,'loss':loss.item(),
                        'interior':loss_int.item(),'boundary':loss_bd.item()})
        if epoch%snapshot_every==0 or epoch in[n_adam,n_adam+n_lbfgs]:
            trajectory.append(snapshot(model))
        if epoch==1 or epoch%100==0:
            print(f'{kind} {phase} epoch {epoch}: {loss.item():.6g}',flush=True)

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(1,n_adam+1):
        if kind=='pinn':
            boundary_iterator=iter(data['training_boundary'])
            for int_mini in data['training_interior']:
                try:
                    bd_mini=next(boundary_iterator)
                except StopIteration:
                    boundary_iterator=iter(data['training_boundary'])
                    bd_mini=next(boundary_iterator)
                optimizer.zero_grad()
                loss=interior_loss(model,int_mini)+lambdabd*boundary_loss(model,bd_mini)
                loss.backward()
                optimizer.step()
        else:
            for batch in data['train_loader']:
                optimizer.zero_grad()
                loss=data_driven_loss(model,batch)
                loss.backward()
                optimizer.step()
        record(epoch,'Adam')

    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=20,history_size=50,line_search_fn='strong_wolfe')
    for step in range(1,n_lbfgs+1):
        def closure():
            optimizer.zero_grad()
            loss=full_loss()[0]
            loss.backward()
            return loss
        optimizer.step(closure)
        record(n_adam+step,'L-BFGS')
    return history,torch.stack(trajectory)
