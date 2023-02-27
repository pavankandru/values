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
)


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

class_dict = {
    1:['Be creative', 'Be curious', 'Have freedom of thought',
       'Be choosing own goals', 'Be independent', 'Have freedom of action',
       'Have privacy', 'Have an exciting life', 'Have a varied life',
       'Be daring', 'Have pleasure', 'Be ambitious', 'Have success',
       'Be capable', 'Be intellectual', 'Be courageous', 'Have influence',
       'Have the right to command', 'Have wealth', 'Have social recognition',
       'Have a good reputation', 'Have a sense of belonging',
       'Have good health', 'Have no debts', 'Be neat and tidy',
       'Have a comfortable life', 'Have a safe country',
       'Have a stable society', 'Be respecting traditions',
       'Be holding religious faith', 'Be compliant', 'Be self-disciplined',
       'Be behaving properly', 'Be polite', 'Be honoring elders', 'Be humble',
       'Have life accepted as is', 'Be helpful', 'Be honest', 'Be forgiving',
       'Have the own family secured', 'Be loving', 'Be responsible',
       'Have loyalty towards friends', 'Have equality', 'Be just',
       'Have a world at peace', 'Be protecting the environment',
       'Have harmony with nature', 'Have a world of beauty', 'Be broadminded',
       'Have the wisdom to accept others', 'Be logical',
       'Have an objective view'],# Headers of file data/level1-labels-training.tsv
    2:['Self-direction: thought', 'Self-direction: action',
       'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
       'Power: resources', 'Face', 'Security: personal', 'Security: societal',
       'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
       'Humility', 'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity'],# Headers of file data/labels-training.tsv
    3:['Self-direction','Power','Security','Conformity','Benevolence','Universalism'],# extra labels from level 2
    4:['Openness to change', 'Self-enhancement', 'Conservation','Self-transcendence'],# Headers of file data/labels-level3.tsv
    5:['Personal focus', 'Social focus'],# Headers of file data/labels-level4a.tsv
    6:['Growth, Anxiety-free','Self-protection, Anxiety avoidance'],# Headers of file data/labels-level4b.tsv
}
class Data(Dataset):
    def __init__(self,tokenizer,split="training") -> None:
        super().__init__()
        df = pd.read_csv('data/arguments-{}.tsv'.format(split),delimiter='\t')
        if split!='test':
            df2 = pd.read_csv('data/labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df2,on=["Argument ID"])
            df1 = pd.read_csv('data/level1-labels-{}.tsv'.format(split),delimiter='\t')
            df = df.merge(df1,on=["Argument ID"])

            df3 = pd.read_csv('data/labels-level3.tsv',delimiter='\t')
            df4 = pd.read_csv('data/labels-level4a.tsv',delimiter='\t')
            df5 = pd.read_csv('data/labels-level4b.tsv',delimiter='\t')
            df = df.merge(df3,on=["Argument ID"],how='left')
            df = df.merge(df4,on=["Argument ID"],how='left')
            df = df.merge(df5,on=["Argument ID"],how='left')
            df = extra_labels(df)
            df = df.fillna(np.random.choice([0, 1]))
            self.labels = df[df.columns.difference(['Argument ID','Conclusion','Stance','Premise'])]

        
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
            return_dict["labels"]=self.labels.iloc[idx]
        return return_dict


def collate_fn(batch):
    unpadded_batch=[]
    for row in batch:
        i,j,k = row['premise'],row['stance'],row['conclusion']
        unpadded_batch.append([1]+i+[2]+j+[2]+k+[2])
    max_len = max([len(i) for i in unpadded_batch])
    x = np.array([i+[0]*(max_len-len(i)) for i in unpadded_batch])
    if 'labels' in batch[0]:
        if row['labels'][class_dict[3]].isna().sum()>0:
            return [torch.LongTensor(x),None,torch.Tensor(np.array([row['labels'][class_dict[i+1]] for row in batch])),None,None,None,None]
        else:
            return [torch.LongTensor(x)] + [torch.Tensor(np.array([row['labels'][class_dict[i+1]] for row in batch])) 
                                    for i in range(6)]
    else:
        return [torch.LongTensor(x)]+[None for i in range(6)]

    

from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl


from pytorch_lightning import Trainer
from sklearn.metrics import classification_report


import pickle



import torch.optim as optim


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

train_loader = DataLoader(Data(tokenizer,"training"),batch_size=16,collate_fn=collate_fn,shuffle=True)
val_loader = DataLoader(Data(tokenizer,"validation"),batch_size=16,collate_fn=collate_fn,)
test_loader = DataLoader(Data(tokenizer,"test"),batch_size=16,collate_fn=collate_fn,)


model = AutoModel.from_pretrained("microsoft/deberta-base")




# define the LightningModule
class MLClassifier(pl.LightningModule):
    def __init__(self, model,dropout=0.5):
        super().__init__()
        self.model=model

        self.classifier1 = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[1]))])
        self.classifier2 = nn.Sequential(*[
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[2]))])
        self.classifier3 = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[3]))])
        self.classifier4 = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[4]))])
        self.classifier5 = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[5]))])
        self.classifier6 = nn.Sequential(*[nn.Dropout(dropout),
                                         nn.Linear(768,128),
                                         nn.SiLU(),
                                         nn.Linear(128,len(class_dict[6]))])

        self.loss_fn1 = nn.MultiLabelSoftMarginLoss()
        self.loss_fn2 = nn.MultiLabelSoftMarginLoss()
        self.loss_fn3 = nn.MultiLabelSoftMarginLoss()
        self.loss_fn4 = nn.MultiLabelSoftMarginLoss()
        self.loss_fn5 = nn.MultiLabelSoftMarginLoss()
        self.loss_fn6 = nn.MultiLabelSoftMarginLoss()
        self.class_names = class_dict[2]
        self.train_res=[]
        self.val_res=[]

    def forward(self,x):
        otp =self.model(x,output_hidden_states=True)['hidden_states']
        
        emb1 = emb2 = otp[-1][:,0,:] #CLS token
        emb3 = otp[-2][:,0,:] #CLS token
        emb4 = otp[-3][:,0,:] #CLS token
        emb5 = otp[-4][:,0,:] #CLS token
        emb6 = otp[-5][:,0,:] #CLS token

        o1 = self.classifier1(emb1)
        o2 = self.classifier2(emb2)
        o3 = self.classifier3(emb2)
        o4 = self.classifier4(emb2)
        o5 = self.classifier5(emb2)
        o6 = self.classifier6(emb2)
        return o1,o2,o3,o4,o5,o6

    def prediction_reducer(self,otps):
        print(len(otps))
        print(otps[0]['predictions'].shape)
        predictions = torch.cat([i['predictions'].detach() for i in otps])
        # predictions = predictions[:,[0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]]
        predictions = torch.sigmoid(predictions)
        if 'labels' in otps[0]:
            labels = torch.cat([i['labels'].detach() for i in otps])
            print(predictions.shape)
            # labels=labels[:,[0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]]
            return predictions,labels
        return predictions

    def training_step(self, batch, batch_idx):
        X,y1,y2,y3,y4,y5,y6 = batch
        o1,o2,o3,o4,o5,o6 = self.forward(X)
        if y1 is not None:
            loss1 = self.loss_fn1(o1, y1)
            loss2 = self.loss_fn2(o2, y2)
            loss3 = self.loss_fn3(o3, y3)
            loss4 = self.loss_fn4(o4, y4)
            loss5 = self.loss_fn5(o5, y5)
            loss6 = self.loss_fn6(o6, y6)
            
            loss = loss1+(5*loss2)+loss3+loss4+loss5+loss6
            loss/=10
        else:
            loss = self.loss_fn2(o2, y2)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.train_res.append({"predictions":o2,"labels":y2})
        return {"loss": loss, "predictions": o2,"labels":y2}

    def training_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(self.train_res)
        labels,predictions = labels.cpu(),1.0*(predictions.cpu()>0.1)
        print(torch.mean(predictions,axis=0))
        self.train_res=[]
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(labels,predictions,target_names=self.class_names,zero_division=1,output_dict=True)
        self.log("Training Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Training Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n****************************Training****************************')
        print(classification_report(labels,predictions,target_names=self.class_names,zero_division=1))

    def validation_step(self, batch,batch_idx):
        X,y1,y2,y3,y4,y5,y6 = batch
        o1,o2,o3,o4,o5,o6 = self.forward(X)
        if y1 is not None:
            loss1 = self.loss_fn1(o1, y1)
            loss2 = self.loss_fn2(o2, y2)
            loss3 = self.loss_fn3(o3, y3)
            loss4 = self.loss_fn4(o4, y4)
            loss5 = self.loss_fn5(o5, y5)
            loss6 = self.loss_fn6(o6, y6)
            
            loss = loss1+(5*loss2)+loss3+loss4+loss5+loss6
            loss/=10
        else:
            loss = self.loss_fn2(o2, y2)
        
        loss = loss1+(5*loss2)+loss3+loss4+loss5+loss6
        # Logging to TensorBoard (if installed) by default
        self.log("Validation loss", loss,on_epoch=True)
        self.val_res.append({"predictions":o2,"labels":y2})
        return {"loss": loss, "predictions": o2,"labels":y2}

    def validation_epoch_end(self, training_step_outputs):
        predictions,labels = self.prediction_reducer(self.val_res)
        labels,predictions = labels.cpu(),1.0*(predictions.cpu()>0.5)
        print(torch.mean(predictions,axis=0))
        self.val_res=[]
        # print("\n\n\nHere\n\n\n",predictions.shape,labels.shape)
        report = classification_report(labels,predictions,target_names=self.class_names,zero_division=1,output_dict=True)
        # lr=self.optimizer.lr_scheduler.get_lr()
        # self.log("Learning Rate", lr,on_epoch=True)
        self.log("Validation Macro F1", report['macro avg']['f1-score'],on_epoch=True)
        self.log("Validation Micro F1", report['micro avg']['f1-score'],on_epoch=True)
        print('\n\n****************************Validation****************************')
        print(classification_report(labels,predictions,target_names=self.class_names,zero_division=1))
    
    def predict_step(self, batch, batch_idx):
        # take average of `self.mc_iteration` iterations
        X,y = batch
        pred = self(X)
        return pred
    
    

    def configure_optimizers(self):
        params = list(self.named_parameters())

        def is_backbone(n): return 'bert' in n

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], 'lr': 1e-5},
            {"params": [p for n, p in params if not is_backbone(n)], 'lr': 1e-3},
        ]
        optimizer = optim.AdamW(grouped_parameters, lr=1e-5)
        # return optimizer
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.3)
        return {"optimizer": optimizer,  "lr_scheduler": lr_scheduler}



# init the autoencoder
clf = MLClassifier(model)

trainer = pl.Trainer( max_epochs=50,accelerator="gpu", devices=1,logger=wandb_logger,)
trainer.fit(model=clf, train_dataloaders=train_loader,val_dataloaders = val_loader)
# trainer.fit(model=clf, train_dataloaders=val_loader,val_dataloaders = val_loader)


# from temperature_scaling import ModelWithTemperature
# scaled_model = ModelWithTemperature(clf)
# scaled_model.set_temperature(val_loader)




# predictions_test = trainer.predict(model=clf,dataloaders=test_loader)
# predictions = trainer.predict(model=clf,dataloaders=val_loader)


# pickle.dump(predictions,open('predictions_val4lr','wb'))
# pickle.dump(predictions_test,open('predictions_test4lr','wb'))