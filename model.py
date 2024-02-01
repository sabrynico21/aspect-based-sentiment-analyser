import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from transformers import BertModel
from utils import AverageValueMeter

class AspectClassificationModel(nn.Module):

    def __init__(self, bert_model_name, num_aspects):
        super(AspectClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_output_dim, num_aspects)

    @classmethod
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  
        # Extract the output embeddings for each token
        token_embeddings = outputs.last_hidden_state  
        # Apply a linear layer to obtain scores for each aspect
        aspect_scores = self.fc(token_embeddings)  
        # Apply softmax to get probabilities
        aspect_probs = F.softmax(aspect_scores, dim=-1)  
        return aspect_probs
    
    @classmethod
    def train_and_save(self, num_epochs, train_loader, val_loader, criterion, optimizer, num_classes,logging):
        #model_path = 'LSTM_epoch_{}.pth'.format(5)  # Adjust the epoch number accordingly
        #model.load_state_dict(torch.load(model_path))
        loader = {
        'train' : train_loader,
        'val' : val_loader
        }
        
        metrics_dict = {
            'train': {
                'loss': AverageValueMeter(),
                'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
                'precision': torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass'),
                'recall': torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
            },
            'val': {
                'loss': AverageValueMeter(),
                'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes),
                'precision': torchmetrics.Precision(num_classes=num_classes, average='macro', task='multiclass'),
                'recall': torchmetrics.Recall(num_classes=num_classes, average='macro', task='multiclass')
            }
        }
        for epoch in range(num_epochs):
            for mode in ['train', 'val']:
                self.train() if mode == 'train' else self.eval()
                metrics = metrics_dict[mode]
                
                metrics['loss'].reset()
                metrics['accuracy'].reset()
                metrics['precision'].reset()
                metrics['recall'].reset()
                with torch.set_grad_enabled(mode=='train'):
                    for inputs, attention_mask, labels in loader[mode]:
        
                        outputs = self(inputs, attention_mask)
                        labels = torch.argmax(labels, dim=2).long()
                        outputs=outputs.permute(0, 2, 1)
                        loss = criterion(outputs, labels)
                        if mode == 'train':
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                        n=inputs.shape[0]
                        _, predicted = torch.max(outputs, 1)
                        
                        mask = (attention_mask != 0)
                        masked_predict = predicted[mask]
                        masked_labels = labels[mask]
            
                        metrics['loss'].add(loss.item(),n)
                        metrics['accuracy'](masked_predict, masked_labels)
                        metrics['precision'](masked_predict, masked_labels)
                        metrics['recall'](masked_predict, masked_labels)
                        print(epoch)
                logging.info(f'{mode.capitalize()} Mode - '
                    f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Loss: {metrics["loss"].compute():.4f}, '
                    f'Accuracy: {metrics["accuracy"].compute():.4f}, '
                    f'Precision: {metrics["precision"].compute():.4f}, '
                    f'Recall: {metrics["recall"].compute():.4f}, ')     
                print(
                    f'{mode.capitalize()} Mode - '
                    f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Loss: {metrics["loss"].compute():.4f}, '
                    f'Accuracy: {metrics["accuracy"].compute():.4f}, '
                    f'Precision: {metrics["precision"].compute():.4f}, '
                    f'Recall: {metrics["recall"].compute():.4f}, '
                )
            torch.save(self.state_dict(), f'NLP_epoch_{epoch+1}.pth')