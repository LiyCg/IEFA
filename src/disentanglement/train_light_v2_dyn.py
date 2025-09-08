'''
+ loads/unloads dataset for each expression variation dataset (b -> f -> e)
+ combines dynamic data (0 -> ROM , temporally linear valued sequence) 
# combines static data but with different magnitude of ROM (s1 1.0 -> s2 0.75 -> s3 0.5)

'''
import time
import sys
import gc
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
                        if i ==0:
                            print("not training Econ for lip contact loss...\n")                        
                    self.autoencoder.exp_encoder.zero_grad()
                    i += 1
                    # compute gradients for other_losses
                    # called backward() again without zeroing the gradients > the new gradients will be added to the current gradients in .grad
                    other_losses.backward()
                    
                    # update all parameters based on the accumulated(added) gradients
                    self.optimizer.step() # all the gradients are accumulated in the same computational graph, so don't need multiple step()s

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
            i += 1
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
    # hparams = disentangle_args()
    hparams = IEFA_args()
    
    # Data Loader
    if not hparams.use_curriculum_training:
        dataset = data_manager.get_dataloader(hparams)
        
    runner = Runner(hparams)
    min_valid_loss = 1000

    print('Training on ' + device_name(hparams.device))
    print('Model: {}'.format(hparams.model_num))

    # Load model
    saved_epoch = 0
    model_path = f"{hparams.save_dir}/{hparams.model_num}.pth"
    if os.path.isfile(model_path):
        print("Keep training ... ")
        runner, saved_epoch = load_model(runner, model_path)
    
    # import pdb;pdb.set_trace()
    # Make path to save log
    if os.path.isdir(hparams.save_tb_dir) == False:
        os.makedirs(hparams.save_tb_dir, exist_ok=True)
    
    # Log
    writer = None
    if hparams.wandb:
        wandb.init(project=f"IEFA",
                   entity="kaist-vml",
                   config={
                       "learning_rate": hparams.lr,
                       "batch_size": hparams.batch_size
                   })
        wandb.define_metric("valid_gt", summary="min")
        wandb.run.name = f"autoencoder-{hparams.model_num}"
    else:
        writer = SummaryWriter(hparams.save_tb_dir)
    
    bshp_dataset = None
    facs_dataset = None
    emot_dataset = None 
    
    global_step = 0

    if saved_epoch != 0:
        global_step = saved_epoch + 1
    else:
        global_step = saved_epoch
    
    # min_valid_loss = float('inf')
    
    if hparams.use_curriculum_training:
        if hparams.use_dynamic:
            phases = [
                ("bshp", hparams.bshp_epochs, hparams.bshpAct_data_dir,   hparams.bshp_dyn_data_dir),
                ("facs", hparams.facs_epochs, hparams.facsAct_data_dir,   hparams.facsAct_dyn_data_dir),
                ("emot", hparams.emot_epochs, hparams.emotion_data_dir,   hparams.emotion_dyn_data_dir),
            ]
            for phase_name, phase_epochs, sta_dir, dyn_dir in phases:
                sta_loader = dyn_loader = None
                # 80% static, 20% dynamic
                n_static = int(0.8 * phase_epochs) if hparams.use_dynamic else phase_epochs

                for e in range(phase_epochs):
                    # decide static vs dynamic
                    if hparams.use_dynamic and e >= n_static:
                        kind = "dynamic"
                        if dyn_loader is None:
                            unload_dataset()
                            hparams.vtx_dtw_path = dyn_dir
                            dyn_loader = data_manager.get_dataloader(hparams)
                        loader = dyn_loader
                    else:
                        kind = "static"
                        if sta_loader is None:
                            unload_dataset()
                            hparams.vtx_dtw_path = sta_dir
                            sta_loader = data_manager.get_dataloader(hparams)
                        loader = sta_loader

                    print(f"[{phase_name.upper()} | Epoch {e+1}/{phase_epochs}] Training on {kind} data")

                    # Run training and evaluation
                    train_loss = runner.run(loader["train"], mode="train")
                    valid_loss = runner.run(loader["valid"], mode="eval")

                    # Prepare logging
                    tag = f"{phase_name}/{kind}"
                       
                    # wandb logging
                    if hparams.wandb:
                        wandb.log({
                            **{f"{tag}/train_{k}": v for k, v in train_loss.items()},
                            **{f"{tag}/valid_{k}": v for k, v in valid_loss.items()}
                        }, step=global_step)
                    # TensorBoard logging
                    else: 
                        writer.add_scalars(tag, {
                            **{f"train_{k}": v for k, v in train_loss.items()},
                            **{f"valid_{k}": v for k, v in valid_loss.items()}
                        }, global_step)
                    # Print summary
                    train_str = " / ".join(f"{k}: {v:.4f}" for k, v in train_loss.items())
                    valid_str = " / ".join(f"{k}: {v:.4f}" for k, v in valid_loss.items())
                    print(f"  Train @ step {global_step}: {train_str}")
                    print(f"  Valid @ step {global_step}: {valid_str}")

                    # Checkpoint every 200 epochs
                    if (e + 1) % 200 == 0 or (e + 1) == phase_epochs:
                        # f"{hparams.save_dir}/{hparams.model_num}.pth"
                        ckpt_path = f"{hparams.save_dir}/{hparams.model_num}.pth"
                        torch.save({
                            'autoencoder_state_dict': runner.autoencoder.state_dict(),
                            'optimizer_state_dict': runner.optimizer.state_dict(),
                            'train_loss': train_loss['loss'],
                            'valid_loss': valid_loss['loss'],
                            'epoch': e + 1,
                            'lr': runner.lr
                        }, ckpt_path)
                        print(f"[Saved checkpoint: {ckpt_path}] and saved final kind = {kind}, epoch = {phase_epochs}")

                    global_step += 1
                    

        else: # not using dynamic just static
            
            for s1 in ["s1", "s2", "s3"]: # 3 stage learning

                if s1 == 's1':
                    continue
                hparams.model_num = "ict_vtx_diffrom_stage_v1" # hardcoded for resume

                saved_epoch = 0
                stage_start_time = time.time()  # 시작 시간 기록

                if s1 == "s1":
                    hparams.bshpAct_data_dir = hparams.bshpAct_data_dir.replace('ict','s1_ict')
                    hparams.facsAct_data_dir = hparams.facsAct_data_dir.replace('ict','s1_ict')
                    hparams.emotion_data_dir = hparams.emotion_data_dir.replace('ict','s1_ict')
                elif s1 == "s2":
                    hparams.bshpAct_data_dir = hparams.bshpAct_data_dir.replace('s1_ict','s2_ict')
                    hparams.facsAct_data_dir = hparams.facsAct_data_dir.replace('s1_ict','s2_ict')
                    hparams.emotion_data_dir = hparams.emotion_data_dir.replace('s1_ict','s2_ict')
                else:
                    hparams.bshpAct_data_dir = hparams.bshpAct_data_dir.replace('s2_ict','s3_ict')
                    hparams.facsAct_data_dir = hparams.facsAct_data_dir.replace('s2_ict','s3_ict')
                    hparams.emotion_data_dir = hparams.emotion_data_dir.replace('s2_ict','s3_ict')

                total_epoch = hparams.bshp_epochs + hparams.facs_epochs + hparams.emot_epochs + 1
                for epoch in range(saved_epoch, total_epoch):
                    if epoch <  hparams.bshp_epochs:
                        if epoch == saved_epoch:
                            unload_dataset()
                            hparams.vtx_dtw_path = hparams.bshpAct_data_dir
                            bshp_dataset = data_manager.get_dataloader(hparams)
                            print(f"training with {s1} bshp\n")
                        train_loss = runner.run(bshp_dataset["train"], 'train')
                        valid_loss = runner.run(bshp_dataset["valid"], 'eval')
                    elif epoch >= hparams.bshp_epochs and epoch < hparams.bshp_epochs + hparams.facs_epochs:
                        if epoch == hparams.bshp_epochs:
                            unload_dataset()
                            hparams.vtx_dtw_path = hparams.facsAct_data_dir
                            facs_dataset = data_manager.get_dataloader(hparams)
                            print(f"training with {s1} facs\n")
                        train_loss = runner.run(facs_dataset["train"], 'train')
                        valid_loss = runner.run(facs_dataset["valid"], 'eval')
                    elif epoch >= hparams.bshp_epochs + hparams.facs_epochs and epoch < hparams.bshp_epochs + hparams.facs_epochs + hparams.emot_epochs:
                        if epoch == hparams.bshp_epochs + hparams.facs_epochs:
                            unload_dataset()
                            hparams.vtx_dtw_path = hparams.emotion_data_dir
                            emot_dataset = data_manager.get_dataloader(hparams)
                            print(f"training with {s1} emotion\n")
                        train_loss = runner.run(emot_dataset["train"], 'train')
                        valid_loss = runner.run(emot_dataset["valid"], 'eval')
                    
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
                    if (epoch + 1) == total_epoch:
                        save_name = f"{hparams.save_dir}/{hparams.model_num}_{s1}_final.pth"
                        elapsed = time.time() - stage_start_time
                        print(f"✅ Finished Stage {s1} in {elapsed//60:.0f}m {elapsed%60:.0f}s\n")
                    else: 
                        save_name = f"{hparams.save_dir}/{hparams.model_num}.pth"
                        
                    if (epoch + 1) % 200 == 0 or (epoch + 1) == total_epoch:
                        torch.save({
                                    'autoencoder_state_dict': runner.autoencoder.state_dict(),
                                    'optimizer_state_dict': runner.optimizer.state_dict(),
                                    'train_loss': train_loss['loss'],
                                    'valid_loss': valid_loss['loss'],
                                    'epoch': epoch,
                                    'lr': runner.lr
                                    #}, f"{hparams.save_dir}/{hparams.model_num}_e{epoch}.pth")
                                    }, save_name)
                        print("[      %d - Saved model %s] " % (epoch, save_name))
    
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
