import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


def extra_labels(df):
    df['Self-direction'] = df[['Self-direction: thought', 'Self-direction: action']].sum(axis=1)
    df['Power'] = df[['Power: dominance','Power: resources']].sum(axis=1)
    df['Security']= df[['Security: personal', 'Security: societal']].sum(axis=1)
    df['Conformity']= df[['Conformity: rules', 'Conformity: interpersonal']].sum(axis=1)
    df['Benevolence']=df[['Benevolence: caring', 'Benevolence: dependability']].sum(axis=1)
    df['Universalism'] =df[['Universalism: concern', 'Universalism: nature','Universalism: tolerance', 'Universalism: objectivity']].sum(axis=1)
    for i in ['Self-direction','Power','Security','Conformity','Benevolence','Universalism']:
        df[i]=(df[i]>0)*1
    return df

class Data(Dataset):
    def __init__(self,tokenizer,split="training") -> None:
        super().__init__()
        df = pd.read_csv('data/arguments-{}.tsv'.format(split),delimiter='\t')
        if split!='test':
            df2 = pd.read_csv('data/labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df2,on=["Argument ID"])
            df = extra_labels(df)
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
    x = np.array([i+[0]*(max_len-len(i)) for i in unpadded_batch])
    
    return torch.LongTensor(x),(torch.Tensor(np.array([row['labels'] for row in batch])) if 'labels' in batch[0] else None)


    

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



wandb_logger = WandbLogger(project="values")



# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model,dropout=0.2):
        super().__init__()
        self.model=model
        self.classifier = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         # Try more 
                                         nn.Linear(128,26)])
        
        weight=torch.Tensor([[0.30705791, 0.15538616, 0.55848939, 0.74826734, 0.13403745,
       0.46994839, 0.58387527, 0.59285797, 0.10154353, 0.15793347,
       0.44809032, 0.16938799, 1.2845256 , 0.60686249, 0.12175598,
       0.28758036, 0.11218564, 0.60686249, 0.34561227, 0.20773999,
       0.32136078, 0.67596578, 0.17676292, 0.40418573, 0.24351562,
       0.17820916]])
        self.loss_fn = nn.MultiLabelSoftMarginLoss(weight=weight)
        self.class_names =['Self-direction: thought', 'Self-direction: action', 'Stimulation',
       'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources',
       'Face', 'Security: personal', 'Security: societal', 'Tradition',
       'Conformity: rules', 'Conformity: interpersonal', 'Humility',
       'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity',
       'Self-direction', 'Power', 'Security', 'Conformity', 'Benevolence',
       'Universalism']

    def forward(self,x):
        emb = self.model(x)['last_hidden_state'][:,0,:]
        o = self.classifier(emb)
        return o

    def prediction_reducer(self,otps):
        predictions = torch.cat([i['predictions'].detach() for i in otps])
        predictions = predictions[:,:-6]
        if 'labels' in otps[0]:
            labels = torch.cat([i['labels'].detach() for i in otps])
            labels=labels[:,:-6]
            return predictions,labels
        return predictions

    def training_step(self, batch, batch_idx):
        X,y = batch
        o = self.forward(X)

        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return {"loss": loss, "predictions": o,"labels":y}

    def training_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names[:20],zero_division=0,output_dict=True)
        self.log("Training Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Training Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Training')
        print(classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names[:20],zero_division=0))

    def validation_step(self, batch,batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X,y = batch
        o = self.forward(X)
        # print("\n\n\nHere\n\n\n",o.shape,y.shape)
        loss = self.loss_fn(o, y)
        # Logging to TensorBoard (if installed) by default
        self.log("Validation loss", loss,on_epoch=True)
        return {"loss": loss, "predictions": o,"labels":y}

    def validation_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(training_step_outputs)
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names[:20],zero_division=0,output_dict=True)
        self.log("Validation Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Validation Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Validation')
        print(classification_report(predictions.cpu()>0.05,labels.cpu(),target_names=self.class_names[:20],zero_division=0))
    
    def predict_step(self, batch, batch_idx):
        # take average of `self.mc_iteration` iterations
        X,y = batch
        pred = self(X)
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


pickle.dump(predictions,open('predictions_val3','wb'))
pickle.dump(predictions_test,open('predictions_test3','wb'))