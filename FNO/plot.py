import matplotlib.pyplot as plt


def plot_results(histories,metrics,output_dir):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,history in histories.items():
        axes[0].plot([row['epoch']for row in history],[row['loss']for row in history],label=name)
    axes[0].set(xlabel='Epoch',ylabel='Training objective',yscale='log')
    axes[0].legend(fontsize=8)
    for name in['unweighted','weighted','finetuned','scratch']:
        if name in metrics:
            key='unknown' if name in['finetuned','scratch']else 'original'
            values=metrics[name][key]
            axes[1].plot([float(t)for t in values],list(values.values()),marker='o',label=f'{name}: {key}')
    axes[1].set(xlabel='Time from initial state',ylabel='Mean relative L2 error')
    axes[1].legend(fontsize=8)
    fig.savefig(output_dir/'training.png',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    values=metrics['fixed']['resolution']
    axes[0].plot([int(n)for n in values],list(values.values()),marker='o')
    axes[0].set(xlabel='Spatial resolution',ylabel='Mean relative L2 error',title='Fixed-time FNO at t=1')
    for name in['weighted','finetuned','scratch']:
        values=metrics[name]['unknown']
        axes[1].plot([float(t)for t in values],list(values.values()),marker='o',label=name)
    axes[1].set(xlabel='Time from initial state',ylabel='Mean relative L2 error',title='Shifted dataset')
    axes[1].legend()
    fig.savefig(output_dir/'evaluation.png',dpi=180)
    plt.close(fig)
