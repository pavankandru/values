import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="values",
                           mode="disabled"
)#


def extra_labels(df):
    df['Self-direction'] = df[['Self-direction: thought', 'Self-direction: action']].sum(axis=1)
    df['Power'] = df[['Power: dominance','Power: resources']].sum(axis=1)
    df['Security']= df[['Security: personal', 'Security: societal']].sum(axis=1)
    df['Conformity']= df[['Conformity: rules', 'Conformity: interpersonal']].sum(axis=1)
    df['Benevolence']=df[['Benevolence: caring', 'Benevolence: dependability']].sum(axis=1)
    df['Universalism'] =df[['Universalism: concern', 'Universalism: nature','Universalism: tolerance', 'Universalism: objectivity']].sum(axis=1)
    for i in ['Self-direction','Power','Security','Conformity','Benevolence','Universalism']:
        df[i]=(df[i]>0)*1

    l3 = pd.read_csv('data/labels-level3.tsv',delimiter='\t')
    # l4a = pd.read_csv('data/labels-level4a.tsv',delimiter='\t')
    # l4b = pd.read_csv('data/labels-level4b.tsv',delimiter='\t')
    df = pd.merge(df,l3,on='Argument ID')
    # df = pd.merge(df,l4a,on='Argument ID')
    # df = pd.merge(df,l4b,on='Argument ID')

    return df

class Data(Dataset):
    def __init__(self,tokenizer,split="training",dummy=False) -> None:
        super().__init__()
        self.dummy=dummy
        if dummy:
            split='training'
        df = pd.read_csv('data/arguments-{}.tsv'.format(split),delimiter='\t')
        if split!='test':
            df2 = pd.read_csv('data/labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df2,on=["Argument ID"])
            df = extra_labels(df)
            self.labels = df[['Self-direction: thought', 'Self-direction: action', 'Stimulation',
       'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources',
       'Face', 'Security: personal', 'Security: societal', 'Tradition',
       'Conformity: rules', 'Conformity: interpersonal', 'Humility',
       'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity',
    #    'Self-direction', 'Power', 'Security', 'Conformity', 'Benevolence',
    #    'Universalism', 'Openness to change', 'Self-enhancement',
    #    'Conservation', 'Self-transcendence',
       ]].values
        
        self.argument_ids  = df['Argument ID'].values
        self.conclusion =df['Conclusion'].values
        self.stance  = df['Stance'].values
        self.premise = df['Premise'].values
        self.tokenizer =tokenizer
        self.split=split
        if dummy:
            print(len(self.labels))

    def __len__(self):
        return len(self.argument_ids) if not self.dummy else 20
    
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


from pytorch_lightning import Trainer
from sklearn.metrics import classification_report


import pickle



import torch.optim as optim

model = AutoModel.from_pretrained("microsoft/deberta-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

train_loader = DataLoader(Data(tokenizer,"training",dummy=True),batch_size=2,collate_fn=collate_fn,shuffle=True)
val_loader = DataLoader(Data(tokenizer,"validation",dummy=True),batch_size=16,collate_fn=collate_fn,)
test_loader = DataLoader(Data(tokenizer,"test"),batch_size=16,collate_fn=collate_fn,)




class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return  x * torch.sigmoid(x)

# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model,dropout=0.2):
        super().__init__()
        self.model=model
        self.classifier = nn.Sequential(*[
                                         SiLU(),
                                         nn.Linear(768,512), 
                                         SiLU(),
                                         nn.Linear(512,256), 
                                         SiLU(),
                                         nn.Linear(256,128), 
                                         SiLU(),
                                         nn.Linear(128,20)])
        
        weight=torch.Tensor([0.20436299, 0.1179275 , 0.33303599, 0.52893951, 0.11640093,
       0.46711541, 0.50658995, 0.48605252, 0.07768442, 0.13272283,
       0.33614847, 0.12151313, 0.83646248, 0.40872598, 0.08794104,
       0.22340302, 0.09152134, 1.12399645, 0.25329498, 0.21282773,
       #0.29102476, 0.78149307, 0.16830348, 0.34922972, 0.22085674,
       #0.18909223, 0.44096777, 0.4231034 , 0.23879121, 0.23047096
       ])
        weight = torch.ones_like(weight)
        self.loss_fn = nn.MultiLabelSoftMarginLoss(weight=weight)
        self.class_names =['Self-direction: thought', 'Self-direction: action', 'Stimulation',
       'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources',
       'Face', 'Security: personal', 'Security: societal', 'Tradition',
       'Conformity: rules', 'Conformity: interpersonal', 'Humility',
       'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity',
    #    'Self-direction', 'Power', 'Security', 'Conformity', 'Benevolence',
    #    'Universalism', 'Openness to change', 'Self-enhancement',
    #    'Conservation', 'Self-transcendence',
        ]

    def forward(self,x):
        emb = self.model(x)['last_hidden_state'][:,0,:]
        o = self.classifier(emb)
        return o

    def prediction_reducer(self,otps):
        predictions = torch.cat([i['predictions'].detach() for i in otps])
        predictions = predictions[:,:20]
        predictions = torch.sigmoid(predictions)
        if 'labels' in otps[0]:
            labels = torch.cat([i['labels'].detach() for i in otps])
            labels=labels[:,:20]
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
        report = classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=self.class_names[:20],zero_division=0,output_dict=True)
        self.log("Training Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Training Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Training')
        a = 1.0*(predictions.cpu().flatten().numpy()>0.5)
        b = 1.0*(labels.cpu().flatten().numpy())
        print(len(a), sum([1 if a[i]==b[i] else 0 for i in range(len(a))])/len(a))
        # print(np.sum(predictions.cpu().flatten().numpy()>0.5==labels.cpu().flatten().numpy()))
        # print(classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=self.class_names[:20],zero_division=0))

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
        report = classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=self.class_names[:20],zero_division=0,output_dict=True)
        self.log("Validation Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Validation Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n Validation')
        a = 1*(predictions.cpu().flatten().numpy()>0.5)
        b = 1*(labels.cpu().flatten().numpy())
        print(len(a), sum([1 if a[i]==b[i] else 0 for i in range(len(a))])/len(a))
        # print(classification_report(predictions.cpu()>0.5,labels.cpu(),target_names=self.class_names[:20],zero_division=0))
    
    def predict_step(self, batch, batch_idx):
        X,y = batch
        pred = self(X)
        return pred
    
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.3)
        # return {"optimizer": optimizer,  "lr_scheduler": lr_scheduler}



# init the autoencoder
clf = MLClassifier(model)
wandb_logger.watch(clf, log="all")

trainer = pl.Trainer( max_epochs=100,accelerator="gpu", devices=1,logger=wandb_logger,)
trainer.fit(model=clf, train_dataloaders=train_loader,val_dataloaders = val_loader)
# trainer.fit(model=clf, train_dataloaders=val_loader,val_dataloaders = val_loader)


# from temperature_scaling import ModelWithTemperature
# scaled_model = ModelWithTemperature(clf)
# scaled_model.set_temperature(val_loader)




# predictions_test = trainer.predict(model=clf,dataloaders=test_loader)
# predictions = trainer.predict(model=clf,dataloaders=val_loader)


# pickle.dump(predictions,open('predictions_val43','wb'))
# pickle.dump(predictions_test,open('predictions_test43','wb'))