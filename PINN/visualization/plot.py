import numpy as np
import matplotlib.pyplot as plt


def plot_data(data,N,K,path):
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,key in zip(axes,['f','u']):
        values=data[key].reshape(N,N)
        v=np.linspace(0,1,N)
        contour=ax.contourf(v,v,values.T,levels=25,cmap='viridis')
        ax.set(xlabel='x',ylabel='y',title=f'{key}, K={K}',aspect='equal')
        fig.colorbar(contour,ax=ax)
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_training(histories,n_adam,path):
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,kind in zip(axes[:2],['pinn','dd']):
        ax.plot([row['epoch']for row in histories[kind]],[row['loss']for row in histories[kind]])
        if n_adam:
            ax.axvline(n_adam,color='gray',linestyle='dashed',label='Start L-BFGS')
            ax.legend()
        ax.set(xlabel='Epoch',ylabel='Training objective',yscale='log',title=kind.upper())
    for key in['interior','boundary']:
        axes[2].plot([row['epoch']for row in histories['pinn']],[row[key]for row in histories['pinn']],label=key)
    axes[2].set(xlabel='Epoch',ylabel='Unweighted MSE',yscale='log',title='PINN loss terms')
    axes[2].legend()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_predictions(data,predictions,path):
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    pos=data['test_pos'].numpy()
    target=data['test_u'].numpy()
    for row,kind in enumerate(['pinn','dd']):
        prediction=predictions[kind].numpy()
        vmin=min(target.min(),prediction.min())
        vmax=max(target.max(),prediction.max())
        for j,(values,title)in enumerate([(target,'Reference'),(prediction,kind.upper()),(np.abs(target-prediction),'Absolute error')]):
            kwargs={'vmin':vmin,'vmax':vmax}if j<2 else{}
            contour=axes[row,j].scatter(pos[:,0],pos[:,1],c=values,s=9,cmap='viridis',**kwargs)
            axes[row,j].set(title=title,xlabel='x',ylabel='y',aspect='equal')
            fig.colorbar(contour,ax=axes[row,j])
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_landscape(landscape,title,path):
    alpha,beta=np.meshgrid(landscape['alpha'],landscape['beta'],indexing='ij')
    log_loss=np.log10(np.maximum(landscape['loss'],1e-30))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    contour=axes[0].contourf(alpha,beta,log_loss,levels=30,cmap='viridis')
    projected=landscape['projected']
    axes[0].plot(projected[:,0],projected[:,1],color='white',linewidth=1,label='Projected training path')
    axes[0].scatter([0],[0],color='red',s=20,label='Final weights')
    axes[0].legend(fontsize=7)
    axes[0].set(xlabel='PC1 displacement',ylabel='PC2 displacement',title=title)
    fig.colorbar(contour,ax=axes[0],label='log10(loss)')
    axes[1].remove()
    ax=fig.add_subplot(1,2,2,projection='3d')
    ax.plot_surface(alpha,beta,log_loss,cmap='viridis',linewidth=0)
    ax.set(xlabel='PC1',ylabel='PC2',zlabel='log10(loss)',title=f"Explained variance: {landscape['explained'].sum():.1%}")
    fig.savefig(path,dpi=180)
    plt.close(fig)


def plot_frequency(metrics,path):
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for kind in['pinn','dd']:
        ax.plot([int(K)for K in metrics],[metrics[K][kind]['interior_relative_l2']for K in metrics],marker='o',label=kind.upper())
    ax.set(xlabel='Maximum sine mode K',ylabel='Interior relative L2 error',yscale='log')
    ax.legend()
    fig.savefig(path,dpi=180)
    plt.close(fig)
