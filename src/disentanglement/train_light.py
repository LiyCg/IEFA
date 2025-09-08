import sys
import gc  # For garbage collection
sys.path.insert(0, '../')
import os
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
sys.path.append('./src/')
from disentanglement import data_manager
from disentanglement.model_AE import AutoEncoder
sys.path.append('../../')
from parser_util import disentangle_args, IEFA_args
# sys.path.append('/input/inyup/IEFA/data/livelink_MEAD')
# import face_model_io

def get_current_stage(epoch, hparams):
    if epoch < hparams.warmup2_epochs:
        return 'warmup2'
    elif epoch < hparams.warmup2_epochs + hparams.warmup_epochs:
        return 'warmup'
    elif epoch < hparams.warmup2_epochs + hparams.warmup_epochs + hparams.bshp_epochs:
        return 'bshp'
    elif epoch < hparams.warmup2_epochs + hparams.warmup_epochs + hparams.bshp_epochs + hparams.facs_epochs:
        return 'facs'
    elif epoch < hparams.warmup2_epochs + hparams.warmup_epochs + hparams.bshp_epochs + hparams.facs_epochs + hparams.emot_epochs:
        return 'emotion'
    return None

def load_dataset(stage, hparams):
    if stage == "warmup2":
        hparams.vtx_dtw_path = hparams.bshpAct_warmup2_data_dir
    elif stage == 'warmup':
        hparams.vtx_dtw_path = hparams.bshpAct_warmup_data_dir
    elif stage == "bshp":
        hparams.vtx_dtw_path = hparams.bshpAct_data_dir
    elif stage == "facs":
        hparams.vtx_dtw_path = hparams.facsAct_data_dir
    elif stage == 'emotion':
        hparams.vtx_dtw_path = hparams.emotion_data_dir
    
    print(f"\n loading {stage} datset...")
    dataset = data_manager.get_dataloader(hparams)
    print(f"✅ {stage} dataset loaded!!")
    return dataset

def unload_dataset():
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 Dataset unloaded, GPU cache cleared")

class Runner(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.lr = hparams.lr
        # if self.hparams.use_lip_contact_loss:
        #     face_model = face_model_io.load_face_model('/input/inyup/ICT-FaceKit/FaceXModel')
        # else: 
        #     face_model = None
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
        if self.hparams.use_lip_contact_loss:
            epoch_loss = {'loss': 0.0, 'cross': 0.0, 'self': 0.0, 'con_tpl': 0.0, 'exp_tpl': 0.0, 'lip_contact':0.0, 'lip_strong':0.0, 'lip_loose':0.0}
        else:
            epoch_loss = {'loss': 0.0, 'cross': 0.0, 'self': 0.0, 'con_tpl': 0.0, 'exp_tpl': 0.0}
        pbar = enumerate(dataloader)
        i = 0 
        for batch, data in pbar:
            vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2 = data
            vtx_c1e1 = vtx_c1e1.to(self.device).float()
            vtx_c2e1 = vtx_c2e1.to(self.device).float()
            vtx_c1e2 = vtx_c1e2.to(self.device).float()
            vtx_c2e2 = vtx_c2e2.to(self.device).float()

            self.optimizer.zero_grad() # clear gradients from the previous step
            # When you pass data through a model (i.e., forward pass), all the operations (matrix multiplications, additions, etc.) performed by the layers of the model are recorded in computational graph
            loss_dict = self.autoencoder(vtx_c1e1, vtx_c2e1, vtx_c1e2, vtx_c2e2)
            
            if self.hparams.use_lip_contact_loss:
                ## lip contact loss version
                lip_contact_loss = loss_dict['lip_contact']
                excluded_keys = ['lip_contact', 'lip_strong', 'lip_loose'] 
                # import pdb;pdb.set_trace()
                # other_losses = sum([v for k, v in loss_dict.items() if k.split('_') != 'lip'])
                filtered_loss_dict = {k: v for k, v in loss_dict.items() if k not in excluded_keys}
                other_losses = sum(filtered_loss_dict.values())

                if mode == 'train':
                    
                    # compute gradients for lip_contact_loss
                    # done by traversing the computational graph backward from the loss node
                    # These gradients are stored in each parameter's .grad attribute
                    lip_contact_loss.backward(retain_graph=True) # retain_graph to not free the gradient graph for second forward pass
                    # restricting it to the decoder
                    if not self.hparams.use_train_con:
                        self.autoencoder.con_encoder.zero_grad()
                        if i == 0:
                            print("not training Econ...\n")                        
                    self.autoencoder.exp_encoder.zero_grad()
                    
                    # compute gradients for other_losses
                    # called backward() again without zeroing the gradients > the new gradients will be added to the current gradients in .grad
                    other_losses.backward()
                    
                    # update all parameters based on the accumulated(added) gradients
                    self.optimizer.step() # all the gradients are accumulated in the same computational graph, so don't need multiple step()s
                    i += 1
                total_loss = lip_contact_loss + other_losses
                epoch_loss['loss'] += vtx_c1e1.size(0) * total_loss.item()
                for k, v in loss_dict.items():
                    # if k == 'lip_strong' or k == 'lip_loose':
                    #     continue
                    epoch_loss[k] += vtx_c1e1.size(0) * v.item()
            else:
                ## original version 
                # loss = sum(loss_dict.values())
                filtered_loss_dict = {k: v for k, v in loss_dict.items() if k not in {'lip_strong', 'lip_loose', 'lip_contact'}}
                loss = sum(filtered_loss_dict.values())
                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()

                epoch_loss['loss'] += vtx_c1e1.size(0) * loss.item()
                for k, v in filtered_loss_dict.items():
                    
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
    hparams = IEFA_args()
    runner = Runner(hparams)
    saved_epoch = 0
    saved_stage = None

    print('Training on ' + device_name(hparams.device))
    print('Model: {}'.format(hparams.model_num))

    # Load model and resume if checkpoint exists
    model_path = f"{hparams.save_dir}/{hparams.model_num}.pth"
    if os.path.isfile(model_path):
        print("🔄 Resuming training from checkpoint...")
        checkpoint = torch.load(model_path)
        runner.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        runner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        saved_stage = checkpoint.get('stage', None)
        print(f"✅ Resumed from epoch {saved_epoch}, stage {saved_stage}")

    # Setup logging
    if not os.path.isdir(hparams.save_tb_dir):
        os.makedirs(hparams.save_tb_dir, exist_ok=True)

    if hparams.wandb:
        wandb.init(project="IEFA",
                   entity="kaist-vml",
                   config={"learning_rate": hparams.lr, "batch_size": hparams.batch_size})
        wandb.run.name = f"autoencoder-{hparams.model_num}"
    else:
        writer = SummaryWriter(hparams.save_tb_dir)

    start_epoch = saved_epoch if saved_epoch != 0 else 0

    # Curriculum Training with Dynamic Dataset Loading
    curriculum_stages = ['warmup2', 'warmup', 'bshp', 'facs', 'emotion']
    stage_epochs = {
        'warmup2': hparams.warmup2_epochs,
        'warmup': hparams.warmup_epochs,
        'bshp': hparams.bshp_epochs,
        'facs': hparams.facs_epochs,
        'emotion': hparams.emot_epochs,
    }

    ## temporary hard coding of stage info cause I don't have 'stage' info yet
    saved_stage = 'warmup2'
    
    for stage in curriculum_stages:
        if stage_epochs[stage] > 0:
            # Skip already completed stages
            if saved_stage and curriculum_stages.index(stage) < curriculum_stages.index(saved_stage):
                continue

            # Load dataset for the stage
            dataset = load_dataset(stage, hparams)
            stage_start_epoch = max(start_epoch, sum(stage_epochs[s] for s in curriculum_stages if curriculum_stages.index(s) < curriculum_stages.index(stage)))
            stage_end_epoch = stage_start_epoch + stage_epochs[stage]

            for epoch in range(stage_start_epoch, stage_end_epoch):
                print(f"\n🛠️ Training Stage: {stage}, Epoch: {epoch + 1}/{stage_end_epoch}")
                
                train_loss = runner.run(dataset["train"], 'train')
                valid_loss = runner.run(dataset["valid"], 'eval')

                # Log losses
                loss_dict = {f"train_{k}": v for k, v in train_loss.items()}
                loss_dict.update({f"valid_{k}": v for k, v in valid_loss.items()})

                if hparams.wandb:
                    wandb.log(loss_dict, step=epoch)
                else:
                    writer.add_scalars('Loss', loss_dict, epoch)

                log = f"[Epoch {epoch + 1}] Train: " + " / ".join(f"{k}: {v:.4f}" for k, v in train_loss.items())
                print(log)
                log = "           Valid: " + " / ".join(f"{k}: {v:.4f}" for k, v in valid_loss.items())
                print(log)

                # Save checkpoint every 200 epochs
                if epoch % 200 == 0:
                    torch.save({
                        'autoencoder_state_dict': runner.autoencoder.state_dict(),
                        'optimizer_state_dict': runner.optimizer.state_dict(),
                        'train_loss': train_loss['loss'],
                        'valid_loss': valid_loss['loss'],
                        'epoch': epoch,
                        'stage': stage,
                        'lr': runner.lr
                    }, model_path)
                    print(f"💾 Saved model at epoch {epoch}, stage {stage}.")

            # Unload dataset after stage completion
            unload_dataset()
        
    else: # original, non curriculum case
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
