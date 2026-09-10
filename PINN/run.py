import json
import sys
from pathlib import Path
import numpy as np
import torch
from.data import generate_data,create_dataloaders
from.model import NN
from.train import fit
from.losses import total_pde_loss,data_driven_loss
from.evaluate import evaluate
from.landscape import loss_landscape
from.plot import plot_data,plot_training,plot_predictions,plot_landscape,plot_frequency


def run(config):
    torch.set_num_threads(config.get('threads',4))
    output_dir=Path(config['output_dir'])
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    metrics={}
    for K in config['K_values']:
        directory=output_dir/f'K{K}'
        directory.mkdir(exist_ok=True)
        raw=generate_data(config['N'],K,config['r'],config['data_seed'])
        np.savez(directory/'data.npz',**raw)
        data=create_dataloaders(raw,config['split_seed'],config['batch_size'])
        plot_data(raw,config['N'],K,directory/'data.png')
        histories={}
        predictions={}
        metrics[str(K)]={}
        for kind in['pinn','dd']:
            print(f'Training {kind}, K={K}',flush=True)
            model=NN(config['model_seed'])
            model.init_xavier()
            history,trajectory=fit(model,data,config['adam_epochs'],config['lbfgs_epochs'],
                                   config['lambdabd'],kind,config['snapshot_every'])
            histories[kind]=history
            metrics[str(K)][kind],predictions[kind]=evaluate(model,data,kind)
            torch.save({'model':model.state_dict(),'trajectory':trajectory,'config':config,'K':K,
                        'u_mean':data['u_mean'],'u_std':data['u_std']},directory/f'{kind}.pt')
            if config['landscape_grid']:
                if kind=='pinn':
                    def loss_fn():
                        return total_pde_loss(model,data['training_interior'].dataset.tensors,
                                              data['training_boundary'].dataset.tensors,config['lambdabd'])
                else:
                    def loss_fn():
                        return data_driven_loss(model,data['train_loader'].dataset.tensors)
                landscape=loss_landscape(model,trajectory,loss_fn,config['landscape_grid'])
                np.savez(directory/f'{kind}_landscape.npz',**landscape)
                plot_landscape(landscape,f'{kind.upper()}, K={K}',directory/f'{kind}_landscape.png')
        (directory/'history.json').write_text(json.dumps(histories,indent=2)+'\n')
        plot_training(histories,config['adam_epochs'],directory/'training.png')
        plot_predictions(data,predictions,directory/'predictions.png')
        (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    plot_frequency(metrics,output_dir/'frequency.png')
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__=='__main__':
    path=Path(sys.argv[1])if len(sys.argv)>1 else Path(__file__).with_name('config.json')
    run(json.loads(path.read_text()))
