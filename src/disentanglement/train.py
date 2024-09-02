import sys
sys.path.insert(0, '../')
import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from disentanglement import data_manager
from disentanglement.model_AE import AutoEncoder
from parser_util import disentangle_args

class Runner(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.lr = hparams.lr
        self.autoencoder = AutoEncoder(hparams)
        self.device = torch.device("cpu")
        # GPU Setting
        if hparams.device > 0:
            torch.cuda.set_device(hparams.device - 1)
            self.device = torch.device("cuda:" + str(hparams.device - 1))
            self.autoencoder.cuda(hparams.device - 1)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.autoencoder.parameters()), lr=self.lr) #TODO: 모든 모델 다 포함되어있는건가?

    def run(self, dataloader, mode='train'):
        self.autoencoder.train() if mode == 'train' else self.autoencoder.eval()
        epoch_loss = {'loss': 0.0, 'cross': 0.0, 'self': 0.0, 'con_tpl': 0.0, 'exp_tpl': 0.0}
        pbar = enumerate(dataloader)
        for batch, data in pbar:
            vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2 = data
            vtx_c1e1 = vtx_c1e1.to(self.device).float()
            vtx_c2e1 = vtx_c2e1.to(self.device).float()
            vtx_c1e2 = vtx_c1e2.to(self.device).float()
            vtx_c2e2 = vtx_c2e2.to(self.device).float()

            self.optimizer.zero_grad()
            loss_dict = self.autoencoder(vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2)
            loss = sum(loss_dict.values())

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            epoch_loss['loss'] += vtx_c1e1.size(0) * loss.item()
            for k, v in loss_dict.items():
                epoch_loss[k] += vtx_c1e1.size(0) * v.item()

        for k, v in epoch_loss.items():
            epoch_loss[k] = v / (len(dataloader.dataset))

        return epoch_loss

def load_model(runner, model_path):
    checkpoint = torch.load(model_path)
    runner.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    runner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    saved_epoch = checkpoint['epoch']
    print("saved epoch: {}".format(saved_epoch))
    print("saved train loss: {}".format(checkpoint['train_loss']))
    print("saved valid loss: {}".format(checkpoint['valid_loss']))
    return runner, saved_epoch

def device_name(device):
    device_name = 'CPU' if device == 0 else 'GPU:' + str(device - 1)
    return device_name

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    hparams = disentangle_args()

    # Data Loader
    dataset = data_manager.get_dataloader(hparams)
    runner = Runner(hparams)
    min_valid_loss = 1000
    saved_epoch = 0

    print('Training on ' + device_name(hparams.device))
    print('Model: {}'.format(hparams.model_num))

    # Load model
    model_path = f"{hparams.save_dir}/{hparams.model_num}.pth"
    if os.path.isfile(model_path):
        print("Keep training ... ")
        runner, saved_epoch = load_model(runner, model_path)

    # Make path to save log
    if os.path.isdir(hparams.save_tb_dir) == False:
        os.makedirs(hparams.save_tb_dir, exist_ok=True)

    # Log
    if hparams.wandb:
        wandb.init(project=f"TeTEC",
                   entity="kaist-vml",
                   config={
                       "learning_rate": hparams.lr,
                       "batch_size": hparams.batch_size
                   })
        wandb.define_metric("valid_gt", summary="min")
        wandb.run.name = f"autoencoder-{hparams.model_num}"
    else:
        writer = SummaryWriter(hparams.save_tb_dir)

    for epoch in range(hparams.epochs):
        epoch += saved_epoch + 1
        # import pdb;pdb.set_trace()
        train_loss = runner.run(dataset["train"], 'train')
        valid_loss = runner.run(dataset["valid"], 'eval')

        loss_dict = {}
        for key in train_loss.keys():
            loss_dict['train_' + key] = train_loss[key]
            loss_dict['valid_' + key] = valid_loss[key]

        if hparams.wandb:
            wandb.log(loss_dict, step=epoch)
        else:
            writer.add_scalars('Loss', loss_dict, epoch)

        log = "[Epoch %d] Train " % (epoch)
        for key, value in train_loss.items():
            log += "%s: %.4f / " % (key, value*100)
        print(log)
        log = "           Valid "
        for key, value in valid_loss.items():
            log += "%s: %.4f / " % (key, value*100)
        print(log)

        # Save
        # if min_valid_loss > valid_loss['loss']:
        #     min_valid_loss = valid_loss['loss']
        if epoch % 200 == 0:
            torch.save({
                        'autoencoder_state_dict': runner.autoencoder.state_dict(),
                        'optimizer_state_dict': runner.optimizer.state_dict(),
                        'train_loss': train_loss['loss'],
                        'valid_loss': valid_loss['loss'],
                        'epoch': epoch,
                        'lr': runner.lr
                        #}, f"{hparams.save_dir}/{hparams.model_num}_e{epoch}.pth")
                        }, f"{hparams.save_dir}/{hparams.model_num}.pth")
            print("[      %d - Saved model %s] " % (epoch, hparams.model_num))
