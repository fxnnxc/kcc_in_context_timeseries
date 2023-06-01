# https://huggingface.co/learn/nlp-course/chapter7/6

import torch 
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator

from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf
import os 
import numpy as np 

class GPTTrainer():
    def __init__(self, 
                 model, 
                 train_dataset, 
                 valid_dataset,
                 num_train_epochs,
                 save_freq,
                 eval_steps,
                 batch_size=4,
                 gradient_accumulation_steps=8,
                 weight_decay=0.1,
                 num_warmup_steps=1_000,
                 device='cuda:0',
                 save_dir="results",
                 track_loss=False, 
                 save_models=False,
                 save_token_loss=False, 
                 is_psb = False, 
                 ):
        
        # ---- Build Params ----
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.num_update_steps_per_epoch = int(len(train_dataset)/batch_size)
        self.num_training_steps = num_train_epochs * self.num_update_steps_per_epoch
        self.num_warmup_steps =   num_warmup_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps 
        self.save_dir = save_dir
        self.device = device
        self.is_psb = is_psb
        
        # ---- Build dataloader, optimzers and accelerators ---
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader  = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False )
        optimizer  = AdamW(self.get_grouped_params(model, weight_decay), lr=5e-4)    
        lr_scheduler = get_scheduler(
                            name="linear",
                            optimizer=optimizer,
                            num_warmup_steps=self.num_warmup_steps,
                            num_training_steps=self.num_training_steps,
                        )
        
        
        # ------ Accelerator -------
        self.accelerator = Accelerator(mixed_precision="fp16", 
                                       gradient_accumulation_steps=gradient_accumulation_steps )
        model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(model, 
                                                                                  optimizer, 
                                                                                  train_dataloader, 
                                                                                  lr_scheduler)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader        
        self.lr_scheduler = lr_scheduler
        self.flags = OmegaConf.create({})
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            print("Please remove the directory first,,,,")
            exit()
            
        self.flags.best_perplexity  = 1e9
        self.flags.results = {
            "running_loss" : [],
            "perplexity" : [],
            "bias_loss" : [],
            "scale_loss" : [],
        }
        self.save_token_loss = save_token_loss
        if self.save_token_loss : 
            self.flags.num_samples = 0
            self.flags.results['token_wise_loss'] = [0 for i in range(4)]
            self.token_loss_for_epoch = np.zeros(shape=(self.num_train_epochs, 512)) #set as the maximum token
        
        OmegaConf.save(self.flags, os.path.join(self.save_dir, "config.yaml"))
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(self.save_dir, save_function=self.accelerator.save)
        
        
    def train(self):
        self.model.train()
        with tqdm(total=self.num_training_steps) as pbar:
            # start of a training
            global_step = 0
            for epoch in range(self.num_train_epochs):
                # start of an epoch
                running_loss = 0 
                pbar.set_description(f"🧪{self.__class__.__name__} | {self.save_dir} | [TRAIN]  ")
                for step, batch in enumerate(self.train_dataloader):
                    # start of an minibatch
                    global_step +=1 
                    pbar.update(1)
                    # for casual loss 
                    with self.accelerator.accumulate(self.model):
                        batch['labels'] = batch['input_ids']
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)
                        # -------------
                        outputs = self.model(**batch, save_token_loss=self.save_token_loss)
                        loss = outputs.loss 
                         
                        if self.save_token_loss:
                            for i, j in enumerate(self.model.saved_token_loss):
                                self.flags.num_samples += self.model.saved_token_loss.size(0)
                                if i >= len(self.flags.results['token_wise_loss']):
                                    self.flags.results['token_wise_loss'] = self.flags.results['token_wise_loss'] +  [0 for _ in range(1+ i - len(self.flags.results['token_wise_loss']))]                    
                                self.flags.results['token_wise_loss'][i] += j.item()
                                self.token_loss_for_epoch[epoch, i] += j.item()
                                    
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()


                        running_loss += loss.item()
                        if global_step % 100 == 0:
                            pbar.set_postfix(
                                {
                                    "lr": self.lr_scheduler.get_last_lr()[0],
                                    "steps": global_step,
                                    "running_loss" : running_loss/(step+1),
                                    "loss/train": loss.item() * self.gradient_accumulation_steps,
                                }
                            )

                    
                    if global_step % (self.num_training_steps//self.save_freq)==0 :
                        path = os.path.join(self.save_dir, f'model_{global_step/self.num_training_steps:.1f}.bin')
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        unwrapped_model.save_pretrained(self.save_dir  , save_function=lambda x,y : torch.save(x, path) )
                        np.save(os.path.join(self.save_dir, 'token_loss_for_epoch.npy'), self.token_loss_for_epoch)
                        
                    
                    if (global_step % (self.eval_steps * self.gradient_accumulation_steps)) == 0:
                        eval_loss, perplexity, scale_loss, bias_loss = self.evaluate()
                        print(f"🌹{self.__class__.__name__} | {self.save_dir} | [EVAL] loss/eval: {eval_loss} | 🌟perplexit: {perplexity}")
                        self.accelerator.wait_for_everyone()
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        # unwrapped_model.save_pretrained(self.save_dir, save_function=self.accelerator.save)
                        self.flags.results.running_loss.append(running_loss/(step+1))
                        self.flags.results.perplexity.append(perplexity)
                        if self.is_psb:
                            self.flags.results.bias_loss.append(bias_loss)
                            self.flags.results.scale_loss.append(scale_loss)
                        OmegaConf.save(self.flags, os.path.join(self.save_dir, "config.yaml"))
                        
                        if perplexity < self.flags.best_perplexity :
                            self.flags.best_perplexity = perplexity
                            path = os.path.join(self.save_dir, f'model_best.bin')
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            unwrapped_model.save_pretrained(self.save_dir  , save_function=lambda x,y : torch.save(x, path) )
                    # end of the minibatch
                # end of the epoch 
                self.token_loss_for_epoch[epoch] = self.token_loss_for_epoch[epoch] / (global_step/(epoch+1))
            # end of the train
            path = os.path.join(self.save_dir, f'model_last.bin')
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.save_dir  , save_function=lambda x,y : torch.save(x, path) )
            np.save(os.path.join(self.save_dir, 'token_loss_for_epoch.npy'), self.token_loss_for_epoch)

        
    def evaluate(self):
        is_training = self.model.training 
        self.model.eval()
        scale_losses = [] 
        ce_losses = [] 
        bias_losses = []
        scale_loss = None
        bias_loss = None 
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                batch['labels'] = batch['input_ids']
                outputs = self.model(**batch)
            if self.is_psb:
                ce_losses.append(outputs.ce_loss)
                bias_losses.append(outputs.bias_loss)
                scale_losses.append(outputs.scale_loss)
            else:
                ce_losses.append(outputs.loss)


        ce_loss = torch.mean(torch.tensor(ce_losses))
        if self.is_psb:
            scale_loss = torch.mean(torch.tensor(scale_losses)).item() if scale_losses else None
            bias_loss = torch.mean(torch.tensor(bias_losses)).item() if scale_losses else None
        try:
            perplexity = torch.exp(ce_loss)
        except OverflowError:
            perplexity = float("inf")

                
        if is_training:
            self.model.train()
        return ce_loss.item(), perplexity.item(), scale_loss, bias_loss
            
    def get_grouped_params(self, model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
        params_with_wd, params_without_wd = [], []
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]
