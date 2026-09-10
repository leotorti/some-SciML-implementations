import numpy as np
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters


def pca_directions(trajectory):
    trajectory=trajectory.double()
    if trajectory.ndim!=2 or trajectory.shape[0]<3:
        raise ValueError('PCA needs at least three training snapshots.')
    centered=trajectory-trajectory.mean(dim=0)
    eigenvalues,eigenvectors=torch.linalg.eigh(centered@centered.T)
    order=torch.argsort(eigenvalues,descending=True)
    eigenvalues=eigenvalues[order].clamp_min(0)
    if eigenvalues[1]<=eigenvalues[0]*1e-10:
        raise ValueError('The training trajectory has fewer than two independent directions.')
    directions=(centered.T@eigenvectors[:,order[:2]])/eigenvalues[:2].sqrt()
    for i in range(2):
        pivot=directions[:,i].abs().argmax()
        directions[:,i]*=directions[pivot,i].sign()
    explained=eigenvalues[:2]/eigenvalues.sum()
    return directions,explained


def loss_landscape(model,trajectory,loss_fn,n_grid=31,padding=0.15):
    if n_grid<3 or padding<0:
        raise ValueError('Use n_grid>=3 and padding>=0.')
    weights=parameters_to_vector(model.parameters()).detach().clone()
    directions,explained=pca_directions(trajectory)
    directions=directions.to(weights)
    projected=(trajectory.to(weights)-weights)@directions
    axes=[]
    for i in range(2):
        low=min(projected[:,i].min().item(),0)
        high=max(projected[:,i].max().item(),0)
        margin=max((high-low)*padding,1e-5)
        axes.append(np.linspace(low-margin,high+margin,n_grid))
    values=np.empty((n_grid,n_grid))
    training=model.training
    try:
        model.eval()
        for i,alpha in enumerate(axes[0]):
            for j,beta in enumerate(axes[1]):
                with torch.no_grad():
                    vector_to_parameters(weights+alpha*directions[:,0]+beta*directions[:,1],model.parameters())
                with torch.enable_grad():
                    values[i,j]=loss_fn().item()
    finally:
        with torch.no_grad():
            vector_to_parameters(weights,model.parameters())
        model.train(training)
    return{'alpha':axes[0],'beta':axes[1],'loss':values,
            'projected':projected.cpu().numpy(),'explained':explained.cpu().numpy(),
            'directions':directions.cpu().numpy()}
