import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report



class Data(Dataset):
    def __init__(self,tokenizer,split="training") -> None:
        super().__init__()
        df = pd.read_csv('data/arguments-{}.tsv'.format(split),delimiter='\t')
        if split!='test':
            df2 = pd.read_csv('data/labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df2,on=["Argument ID"])
            self.labels = df[df.columns.difference(['Argument ID','Conclusion','Stance','Premise'])].values
        
        self.argument_ids  = df['Argument ID'].values
        self.conclusion =df['Conclusion'].values
        self.stance  = df['Stance'].values
        self.premise = df['Premise'].values
        self.tokenizer =tokenizer
        self.split=split

    def __len__(self):
        return len(self.argument_ids)
    
    def __getitem__(self,idx):
        return_dict =  {"premise":self.tokenizer(self.premise[idx],add_special_tokens=False)['input_ids'],
                "stance":self.tokenizer(self.stance[idx],add_special_tokens=False)['input_ids'],
                "conclusion":self.tokenizer(self.conclusion[idx],add_special_tokens=False)['input_ids'],
                }
        if self.split!='test':
            return_dict["labels"]=self.labels[idx]
        return return_dict


def collate_fn(batch):
    unpadded_batch=[]
    for row in batch:
        i,j,k = row['premise'],row['stance'],row['conclusion']
        unpadded_batch.append([1]+i+[2]+j+[2]+k+[2])
    max_len = max([len(i) for i in unpadded_batch])
    x = [i+[0]*(max_len-len(i)) for i in unpadded_batch]
    
    return torch.LongTensor(x),(torch.Tensor([row['labels'] for row in batch]) if 'labels' in batch[0] else None)



    

