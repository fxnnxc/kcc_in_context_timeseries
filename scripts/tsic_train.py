import os
import sys
sys.path.append(".")
from kcc.models.tsic import SAILGPT2LMHeadModel
from kcc.datasets import TimeseriesLMDataset
from kcc.trainers.gpt_trainer import GPTTrainer
import fire

def main(
    # data and tokenizer parameters
    # model parameters
    data='electricity',
    n_layer = 0,
    n_head = 0,
    attention_only = True,
    n_prototype=25,
    patch_length=16,
    overlap=True,
    batch_size = 128,
    context_size = 10,
    device = "cuda:0",
    save_freq=10,
    num_train_epochs = 1000,  
    gradient_accumulation_steps=8,
    eval_steps=100,
):
    root_path = f"./data/{data}"
    overlap_ = 'over' if overlap else 'nono'
    attn_ = 'only' if attention_only else  'also' 
    processed_data_file = f"processed_data_{n_prototype}_len_{patch_length}_algo_kmean.pkl"
    
    save_dir = f"./data/hub/{data}-{overlap_}-{patch_length}-{n_prototype}-{context_size}/tsic-{n_layer}-{n_head}-{attn_}"
    
    dataset = TimeseriesLMDataset(data_file=os.path.join(root_path, processed_data_file),
                                  context_size=context_size,
                                  type="train")
    
    model = SAILGPT2LMHeadModel(n_layer=n_layer,
                            n_head=n_head,
                            n_prototype=n_prototype,
                            attention_only=attention_only)
    
    trainer = GPTTrainer(model=model,
                         train_dataset=dataset,
                         num_train_epochs=num_train_epochs,
                         save_freq=save_freq,
                         valid_dataset=dataset,
                         eval_steps=eval_steps,
                         batch_size=batch_size,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         device=device,
                         save_dir=save_dir,
                         save_token_loss=True)
    
    trainer.train()
        

if __name__ == "__main__":
    
    fire.Fire(main)


    