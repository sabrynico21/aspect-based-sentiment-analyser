import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from transformers import BertModel
from .utils import AverageValueMeter

class ClassificationModel(nn.Module):

    def __init__(self, bert_model_name, num_aspects, num_opinions):
        super(ClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size
        self.fc1 = nn.Linear(bert_output_dim, num_aspects)
        self.fc2 = nn.Linear(bert_output_dim, num_opinions)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        aspect_scores = self.fc1(token_embeddings)
        aspect_probs = F.softmax(aspect_scores, dim=-1)  

        opinion_scores = self.fc2(token_embeddings)
        opinion_probs = F.softmax(opinion_scores, dim=-1) 

        return aspect_probs, opinion_probs
    
    def train_and_save(self, num_epochs, train_loader, val_loader, criterion1, criterion2, optimizer1, optimizer2, num_aspects, num_opinions, logging):
        model_path = 'NLP_epoch_{}.pth'.format(3)  
        self.load_state_dict(torch.load(model_path))
        loader = {
        'train' : train_loader,
        'val' : val_loader
        }
        
        metrics_dict = {
            'train': {
                'loss1': AverageValueMeter(),
                'loss2': AverageValueMeter(),
                'accuracy1': torchmetrics.Accuracy(task='multiclass', num_classes=num_aspects),
                'accuracy2': torchmetrics.Accuracy(task='multiclass', num_classes=num_opinions),
                'precision1': torchmetrics.Precision(num_classes=num_aspects, average='macro', task='multiclass'),
                'precision2': torchmetrics.Precision(num_classes=num_opinions, average='macro', task='multiclass'),
                'recall1': torchmetrics.Recall(num_classes=num_aspects, average='macro', task='multiclass'),
                'recall2': torchmetrics.Recall(num_classes=num_opinions, average='macro', task='multiclass')
            },
            'val': {
                'loss1': AverageValueMeter(),
                'loss2': AverageValueMeter(),
                'accuracy1': torchmetrics.Accuracy(task='multiclass', num_classes=num_aspects),
                'accuracy2': torchmetrics.Accuracy(task='multiclass', num_classes=num_opinions),
                'precision1': torchmetrics.Precision(num_classes=num_aspects, average='macro', task='multiclass'),
                'precision2': torchmetrics.Precision(num_classes=num_opinions, average='macro', task='multiclass'),
                'recall1': torchmetrics.Recall(num_classes=num_aspects, average='macro', task='multiclass'),
                'recall2': torchmetrics.Recall(num_classes=num_opinions, average='macro', task='multiclass')
            }
        }
        for epoch in range(num_epochs):
            for mode in ['train', 'val']:
                self.train() if mode == 'train' else self.eval()
                metrics = metrics_dict[mode]
                
                metrics['loss1'].reset()
                metrics['loss2'].reset()
                metrics['accuracy1'].reset()
                metrics['accuracy2'].reset()
                metrics['precision1'].reset()
                metrics['precision2'].reset()
                metrics['recall1'].reset()
                metrics['recall2'].reset()
                with torch.set_grad_enabled(mode=='train'):
                    for inputs, attention_mask, labels1, labels2 in loader[mode]:
        
                        outputs1, outputs2 = self(inputs, attention_mask)

                        labels1 = torch.argmax(labels1, dim=2).long()
                        labels2 = torch.argmax(labels2, dim=2).long()
                        
                        outputs1=outputs1.permute(0, 2, 1)
                        outputs2=outputs2.permute(0, 2, 1)
                        #print(outputs1)
                        
                        loss1 = criterion1(outputs1, labels1)
                        loss2 = criterion2(outputs2, labels2)

                        if mode == 'train':
                            optimizer1.zero_grad()
                            loss1.backward()
                            optimizer1.step()
                            #optimizer1.zero_grad()
                            optimizer2.zero_grad()
                            loss2.backward()
                            optimizer2.step()
                            

                        _, predicted1 = torch.max(outputs1, 1)
                        _, predicted2 = torch.max(outputs2, 1)
                        #print(predicted1)
                        #print(labels1)
                        mask = (attention_mask != 0)
                        
                        metrics['loss1'].add(loss1.item(),inputs.shape[0])
                        metrics['loss2'].add(loss2.item(),inputs.shape[0])
                        metrics['accuracy1'](predicted1[mask], labels1[mask])
                        metrics['accuracy2'](predicted2[mask], labels2[mask])
                        metrics['precision1'](predicted1[mask], labels1[mask])
                        metrics['precision2'](predicted2[mask], labels2[mask])
                        metrics['recall1'](predicted1[mask], labels1[mask])
                        metrics['recall2'](predicted2[mask], labels2[mask])
                        #print(epoch)
                        print(loss1.item(), loss2.item())
                logging.info(f'{mode.capitalize()} Mode - '
                            f'Epoch [{epoch + 21}/{num_epochs + 10}], '
                            f'Loss Task 1: {metrics["loss1"].compute():.4f}, '
                            f'Loss Task 2: {metrics["loss2"].compute():.4f}, '
                            f'Accuracy Task 1: {metrics["accuracy1"].compute():.4f}, '
                            f'Accuracy Task 2: {metrics["accuracy2"].compute():.4f}, '
                            f'Precision Task 1: {metrics["precision1"].compute():.4f}, '
                            f'Precision Task 2: {metrics["precision2"].compute():.4f}, '
                            f'Recall Task 1: {metrics["recall1"].compute():.4f}, '
                            f'Recall Task 2: {metrics["recall2"].compute():.4f}, ')     
                print(
                    f'{mode.capitalize()} Mode - '
                    f'Epoch [{epoch + 21}/{num_epochs + 10}], '
                    f'Loss: {metrics["loss1"].compute():.4f}, '
                    f'Accuracy: {metrics["accuracy1"].compute():.4f}, '
                    f'Precision: {metrics["precision1"].compute():.4f}, '
                    f'Recall: {metrics["recall1"].compute():.4f}, '
                )
            torch.save(self.state_dict(), f'NLP_epoch_{epoch+1}.pth')

    def predict(self, num_aspects, num_opinions, criterion_aspects, criterion_opinions, test_data, model_path):
        predicted_dict = { "aspect": torch.tensor([]),
                      "opinion": torch.tensor([])}
        metrics_dict = {
            'aspect': {
                'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_aspects),
                'precision': torchmetrics.Precision(num_classes=num_aspects, average='macro', task='multiclass'),
                'recall': torchmetrics.Recall(num_classes=num_aspects, average='macro', task='multiclass')
            },
            'opinion': {
                'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_opinions),
                'precision': torchmetrics.Precision(num_classes=num_opinions, average='macro', task='multiclass'),
                'recall': torchmetrics.Recall(num_classes=num_opinions, average='macro', task='multiclass')
             }
        }
        loss = {}
        criterion = {"aspect" : criterion_aspects,
                     "opinion" : criterion_opinions}
        #model_path = 'NLP_epoch_{}.pth'.format(3)  
        self.load_state_dict(torch.load(model_path))
        self.eval()
        with torch.no_grad():  
            for inputs, attention_mask, labels1, labels2 in test_data:
                outputs1, outputs2 = self(inputs, attention_mask)

                for type in ["aspect","opinion"]:
                    metrics = metrics_dict[type]
                    labels = torch.argmax(labels1 if type=="aspect" else labels2, dim=2).long()
                    outputs = (outputs1 if type=="aspect" else outputs2).permute(0, 2, 1)
                    loss[type] = criterion[type](outputs, labels)
                    _ , predicted = torch.max(outputs, 1)
                   
                    predicted_dict[type] = torch.cat([predicted_dict[type], predicted], dim=0)
                    mask = (attention_mask != 0)

                    metrics["accuracy"](predicted[mask], labels[mask])
                    metrics["precision"](predicted[mask], labels[mask])
                    metrics["recall"](predicted[mask], labels[mask])
                            
                    print( f'Loss: {loss[type]:.4f}, '
                        f'Accuracy: {metrics["accuracy"].compute():.4f}, '
                        f'Precision: {metrics["precision"].compute():.4f}, '
                        f'Recall: {metrics["recall"].compute():.4f}, ', '\n'
                        )
        return predicted_dict
            