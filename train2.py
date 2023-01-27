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
    unpadded_batch=[[],[],[]]
    for row in batch:
        i,j,k = row['premise'],row['stance'],row['conclusion']
        unpadded_batch[0].append(i)
        unpadded_batch[1].append(j)
        unpadded_batch[2].append(k)
    max_len = [max([len(i) for i in unpadded_batch[j]]) for j in range(3)]
    x = np.array([i+[0]*(max_len[0]-len(i)) for i in unpadded_batch[0]])
    y = np.array([i+[0]*(max_len[1]-len(i)) for i in unpadded_batch[1]])
    z = np.array([i+[0]*(max_len[2]-len(i)) for i in unpadded_batch[2]])
    
    return torch.LongTensor(x),torch.LongTensor(y),torch.LongTensor(z),(torch.Tensor(np.array([row['labels'] for row in batch])) if 'labels' in batch[0] else None)



    

from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from sklearn.metrics import classification_report


import pickle



import torch.optim as optim

model = AutoModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

train_loader = DataLoader(Data(tokenizer,"training"),batch_size=16,collate_fn=collate_fn,shuffle=True)
val_loader = DataLoader(Data(tokenizer,"validation"),batch_size=16,collate_fn=collate_fn,)
test_loader = DataLoader(Data(tokenizer,"test"),batch_size=16,collate_fn=collate_fn,)



wandb_logger = WandbLogger(project="values",)



# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model,dropout=0.3):
        super().__init__()
        self.model=model
        self.classifier = nn.Sequential(*[
                                         nn.Linear(3*768,128),
                                         nn.ReLU(),# Try more 
                                         nn.Linear(128,20)])
        
        weight=torch.Tensor([0.0278, 0.0197, 0.1111, 0.1596, 0.0182, 0.0450, 0.0439, 0.0719, 0.0137,
        0.0159, 0.0483, 0.0233, 0.1326, 0.0695, 0.0206, 0.0341, 0.0132, 0.0643,
        0.0413, 0.0260])
        self.loss_fn = nn.MultiLabelSoftMarginLoss(weight=weight)
        self.class_names =['Achievement', 'Benevolence: caring', 'Benevolence: dependability',
       'Conformity: interpersonal', 'Conformity: rules', 'Face', 'Hedonism',
       'Humility', 'Power: dominance', 'Power: resources',
       'Security: personal', 'Security: societal', 'Self-direction: action',
       'Self-direction: thought', 'Stimulation', 'Tradition',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: objectivity', 'Universalism: tolerance']

    def forward(self,x,k,l):
        emb1 = self.model(x)['last_hidden_state'][:,0,:]
        emb2 = self.model(k)['last_hidden_state'][:,0,:]
        emb3 = self.model(l)['last_hidden_state'][:,0,:]
        emb = torch.cat([emb1,emb2,emb3],dim=1)
        o = self.classifier(emb)
        return o

    def prediction_reducer(self,otps):
        predictions = torch.cat([i['predictions'].detach() for i in otps])
        if 'labels' in otps[0]:
            labels = torch.cat([i['labels'].detach() for i in otps])
            return predictions,labels
        return predictions

    def training_step(self, batch, batch_idx):
        X,k,l,y = batch
        o = self.forward(X,k,l)
        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        # print("\n\n\nHere\n\n\n",o.shape)
        return {"loss": loss, "predictions": o,"labels":y}

    def training_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        report = classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names,zero_division=0,output_dict=True)
        self.log("Training Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Training Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Training')
        print(classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names,zero_division=0))

    def validation_step(self, batch,bxatch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X,k,l,y = batch
        o = self.forward(X,k,l)
        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        print("\n\n\nHere\n\n\n",o.shape,y.shape)
        self.log("Validation loss", loss,on_epoch=True)
        # print("\n\n\nHere\n\n\n",o.shape)

        return {"loss": loss, "predictions": o,"labels":y}

    def validation_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        report = classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names,zero_division=0,output_dict=True)
        self.log("Validation Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Validation Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Validation')
        print(classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names,zero_division=0))
    
    def predict_step(self, batch, batch_idx):
        # take average of `self.mc_iteration` iterations
        X,k,l,y = batch
        pred = self(X,k,l)
        return pred
    
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.3)
        # return {"optimizer": optimizer,  "lr_scheduler": lr_scheduler}




# init the autoencoder
clf = MLClassifier(model)

trainer = pl.Trainer( max_epochs=50,accelerator="gpu", devices=1,logger=wandb_logger,)
trainer.fit(model=clf, train_dataloaders=train_loader,val_dataloaders = val_loader)
# trainer.fit(model=clf, train_dataloaders=val_loader,val_dataloaders = val_loader)


# from temperature_scaling import ModelWithTemperature
# scaled_model = ModelWithTemperature(clf)
# scaled_model.set_temperature(val_loader)




predictions_test = trainer.predict(model=clf,dataloaders=test_loader)
predictions = trainer.predict(model=clf,dataloaders=val_loader)


pickle.dump(predictions,open('predictions_val2','wb'))
pickle.dump(predictions_test,open('predictions_test2','wb'))