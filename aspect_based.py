import spacy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
#from lstm import AspectClassificationModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchmetrics
#from torchtext.vocab import GloVe
import torch.nn.functional as F
import logging

class AspectClassificationModel(nn.Module):
    def __init__(self, bert_model_name, num_aspects):
        super(AspectClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_output_dim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_output_dim, num_aspects)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)  
        # Extract the output embeddings for each token
        token_embeddings = outputs.last_hidden_state  
        # Apply a linear layer to obtain scores for each aspect
        aspect_scores = self.fc(token_embeddings)  
        # Apply softmax to get probabilities
        aspect_probs = F.softmax(aspect_scores, dim=-1)  
        return aspect_probs

def read_json_file(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data_list

def compare_values(target, tokens):
    for token in tokens:   
        if target.value == token:
            return "True"
    return "False"

class AverageValueMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, value, num):
        self.sum += value * num
        self.num += num

    def compute(self):
        try:
            return self.sum/self.num
        except:
            return None
# def prova():
#     # Load the English language model
#     nlp = spacy.load("en_core_web_sm")

#     # Example sentence
#     text = "the food was well prepared and the service impecable"

#     # Process the text with spaCy
#     doc = nlp(text)

#     # Print the dependency relations
#     for token in doc:
#         print(f"{token.text}: {token.dep_} -> {token.head.text}")

#     # Extract aspects based on specific dependencies
#     aspects = [token.text for token in doc if token.dep_ in ["nsubj", "attr", "dobj", "amod", "compound"]]
#     print("Extracted Aspects:", aspects)

aspect_mapping = {'VOID' : 0,
                'RESTAURANT': 1,
                'FOOD': 2,
                'DRINKS': 3,
                'AMBIENCE': 4,
                'SERVICE': 5,
                'LOCATION': 6}

opinion_mapping = {'VOID' : 0,
                'GENERAL': 1,
                'PRICES': 2,
                'QUALITY': 3,
                'STYLE_OPTIONS': 4,
                'MISCELLANEOUS': 5}

class Token:
    def __init__(self, value=None, text=None, position=None):
        self._value = value
        self._text = text
        self._position = position

    @property
    def value(self):
        return self._value
    
    @property
    def text(self):
        return self._text
    
    @property
    def position(self):
        return self._position
    
    # @labels.setter
    # def labels(self, labels):
    #     #self.label.append(labels)
    #     self._labels.append(labels)
    
class Label:
    def __init__(self, aspect_pos=None, opinion_pos=None, polarity=None, category1=None, category2=None):
        self._aspect_pos = aspect_pos
        self._opinion_pos = opinion_pos 
        self._polarity = polarity 
        self._category_aspect = category1
        self._category_opinion = category2

    @property
    def aspect_pos(self):
        return self._aspect_pos
    
    @property
    def opinion_pos(self):
        return self._opinion_pos
    
    @property
    def polarity(self):
        return self._polarity
    
    @property
    def category_aspect(self):
        return self._category_aspect
    
    @property
    def category_opinion(self):
        return self._category_opinion

class TextWithLabels:
    def __init__(self, text, tokens=None, labels=None, text_aspect=None, attention_mask=None):
        self._text = str(text)
        self._text_tokens = tokens if tokens is not None else []
        self._text_labels = labels if labels is not None else []
        #self._text_aspect= text_aspect if text_aspect is not None else []
        #self._attention_mask = attention_mask if attention_mask is not None else []
    
    @property
    def text(self):
        return self._text
    
    @property
    def text_tokens(self):
        #return [token for token in self.text_tokens]
        return self._text_tokens
    
    # @property
    # def attention_mask(self):
    #     return self._attention_mask
    
    @property
    def text_labels(self):
        return self._text_labels

    @text_tokens.setter
    def text_tokens(self, tokens):
        self._text_tokens = tokens

    @text_labels.setter
    def text_labels(self, labels):
        self._text_labels.append(labels)

    def has_label(self):
        return self._text_labels != []

    def compute_one_hot_format(self, num_aspects, num_opinions):
        text_aspect = [aspect_mapping["VOID"]] * len(self.text_tokens)
        text_opinion = [opinion_mapping["VOID"]] * len(self.text_tokens)
        for label in self.text_labels:
            for pos in label.aspect_pos:
                text_aspect[pos] = label.category_aspect
            for pos in label.opinion_pos:
                text_opinion[pos] = label.category_opinion
        text_aspect=torch.eye(num_aspects)[text_aspect]
        text_opinion=torch.eye(num_opinions)[text_opinion]
        return text_aspect, text_opinion
        
    # def compute_attention_mask(self):
    #     attention_mask = []
    #     for token in self._text_tokens:
    #         attention_mask.append((1 if bool(token.value) else 0))
    #     return attention_mask
    
class TextWithLabelsContainer:

    def __init__(self, texts):
        self._instances = []
        for text in texts:
            self._instances.append(TextWithLabels(text))
        self._len = len(texts)
        self._aspect_label = []
        self._opinion_label = []
        self._attention_mask = []

    @property
    def len(self):
        return self._len
            
    def get_texts(self):
        return [instance.text for instance in self._instances]
    
    def get_tokens(self, pos):
        #return [torch.tensor([token.value for token in instance.text_tokens]) for instance in self._instances]
         return torch.tensor([token.value for token in self._instances[pos].text_tokens], dtype=torch.long)
    def set_text_tokens(self, sentence, pos, tokenizer):
        tokens = [Token(text = tokenizer.decode(token), 
                        value=token.item(),
                        position=position) 
                        for position, token in enumerate(sentence)]
        self._instances[pos].text_tokens = tokens 

    def set_text_labels(self, labels, pos, tokenizer):
        for entry in labels:
            aspect_pos = []
            opinion_pos = []
            words_aspect = entry["aspect"]
            words_opinion = entry["opinion"]
            sentiment = entry["polarity"]
            category = entry["category"].split("#") 
            tokens_aspect = tokenizer(words_aspect)["input_ids"][1:-1]
            #print(tokens_aspect)
            tokens_opinion = tokenizer(words_opinion)["input_ids"][1:-1]
            #print(tokens_opinion)
            for token in self._instances[pos].text_tokens:
                if compare_values(token, tokens_aspect) == "True":
                    aspect_pos.append(token.position)
                if compare_values(token, tokens_opinion) == "True":
                    opinion_pos.append(token.position)
            self._instances[pos].text_labels = Label(aspect_pos = aspect_pos, opinion_pos = opinion_pos, polarity = sentiment, category1=aspect_mapping[category[0]], category2=opinion_mapping[category[1]])
            # #print([label.aspect_pos for label in self._instances[pos].text_labels])
            #print(aspect_pos)

    def compute_one_hot_format(self, pos, num_aspects, num_opinions):
        aspect, opinion = self._instances[pos].compute_one_hot_format(num_aspects, num_opinions)
        self._aspect_label.append(aspect)
        self._opinion_label.append(opinion)

    def set_attention_mask(self, attention_mask):
        #attention_mask = self._instances[pos].compute_attention_mask()
        self._attention_mask.append(attention_mask)
    
    def get_aspect_label(self, pos):
        return self._aspect_label[pos]
        
    def get_opinion_label(self, pos):
        return self._opinion_label[pos]

    def get_attention_mask(self, pos):
        return self._attention_mask[pos]
    
    def get_text_labels(self, pos):
        return self._instances[pos].text_labels

def train_model(num_epochs, train_loader, val_loader, criterion, optimizer, num_classes, model,logging):
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
    # avg_metrics_dict = {
    #     'train': {
    #         'loss': AverageValueMeter(),
    #         'accuracy': AverageValueMeter(),
    #         'precision': AverageValueMeter(),
    #         'recall': AverageValueMeter()
    #     },
    #     'val': {
    #         'loss': AverageValueMeter(),
    #         'accuracy': AverageValueMeter(),
    #         'precision': AverageValueMeter(),
    #         'recall': AverageValueMeter()
    #     }
    # }
    for epoch in range(num_epochs):
        for mode in ['train', 'val']:
            model.train() if mode == 'train' else model.eval()
            metrics = metrics_dict[mode]
            #avg_metrics=avg_metrics_dict[mode]
            
            metrics['loss'].reset()
            metrics['accuracy'].reset()
            metrics['precision'].reset()
            metrics['recall'].reset()
            #for param in model.parameters():
             #   param.requires_grad = (mode == 'train')
            with torch.set_grad_enabled(mode=='train'):
                for inputs, attention_mask, labels in loader[mode]:
    
                    outputs = model(inputs, attention_mask)
                    labels = torch.argmax(labels, dim=2).long()
                    outputs=outputs.permute(0, 2, 1)
                    #print(labels.shape)
                    #print(outputs.shape)
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
                    # print(masked_predict)
                    # print(masked_labels)
                    metrics['loss'].add(loss.item(),n)
                    metrics['accuracy'](masked_predict, masked_labels)
                    metrics['precision'](masked_predict, masked_labels)
                    metrics['recall'](masked_predict, masked_labels)
                    print(epoch)
                    # print(metrics["accuracy"].compute())
                    # print(metrics["precision"].compute())
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
               # f'IoU: {metrics["iou"].compute():.4f}'
            )
            #avg_metrics['iou'].add(metrics["iou"].compute(),1)
            # avg_metrics['loss'].add(metrics["loss"].compute(),1)
            # avg_metrics['accuracy'].add(metrics["accuracy"].compute(),1)
            # avg_metrics['precision'].add(metrics["precision"].compute(),1)
            # avg_metrics['recall'].add(metrics["recall"].compute(),1)

        torch.save(model.state_dict(), f'NLP_epoch_{epoch+1}.pth')
    # for mode in ['train', 'val']:
    #     avg_metrics = avg_metrics_dict[mode]
    #     print(f'Avg {mode.capitalize()} Metrics - '
    #           f'Loss: {avg_metrics["loss"].compute():.4f}, '
    #           f'Accuracy: {avg_metrics["accuracy"].compute():.4f}, '
    #           f'Precision: {avg_metrics["precision"].compute():.4f}, '
    #           f'Recall: {avg_metrics["recall"].compute():.4f}, '
    #          )
        
def calculate_class_weights(labels):
    concatenated_labels = torch.cat(labels, dim=0)
    _, class_counts = torch.unique(concatenated_labels, return_counts=True)
    class_frequencies = class_counts.float() / class_counts.sum().float()
    class_weights = 1.0 / class_frequencies
    class_weights /= class_weights.sum()
    return class_weights



def main():
    dataset = {}
    for set in ['train', 'val']:
        file_path =f'C:/Users/sabry/Downloads/rest16_quad_{set}.tsv.jsonl'
        dicts_data = read_json_file(file_path)
        dataset[set] = TextWithLabelsContainer([entry['text'] for entry in dicts_data])

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_inputs = tokenizer(dataset[set].get_texts(), return_tensors="pt", padding=True)
        sentences_tokens = tokenized_inputs["input_ids"]
        sentences_attention_mask = tokenized_inputs["attention_mask"]
        num_aspects = len(aspect_mapping)
        num_opinions = len(opinion_mapping)
        
        for i, sentence in enumerate(sentences_tokens):
            dataset[set].set_text_tokens(sentence, i, tokenizer)
            dataset[set].set_text_labels(dicts_data[i]["labels"], i, tokenizer)
            dataset[set].set_attention_mask(sentences_attention_mask[i])
            dataset[set].compute_one_hot_format(i, num_aspects, num_opinions)

    # aspect = dataset["train"].get_aspect_label()
    # print(aspect[0])   
    # opinion = dataset["train"].get_opinion_label()
    # print(opinion[0])
    # #print(torch.stack(aspect).shape)
    # print(dataset["train"].get_attention_mask()[0])
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = AspectClassificationModel(bert_model_name='bert-large-uncased', num_aspects=num_aspects) 
    model = AspectClassificationModel(bert_model_name='bert-base-uncased', num_aspects=num_aspects)
    for param in model.bert.parameters():
        param.requires_grad = False
    
    train_data = [(dataset["train"].get_tokens(pos),
                  dataset["train"].get_attention_mask(pos),
                  dataset["train"].get_aspect_label(pos)) for pos in range(dataset["train"].len)]
        
    val_data = [(dataset["val"].get_tokens(pos),
                  dataset["val"].get_attention_mask(pos),
                  dataset["val"].get_aspect_label(pos)) for pos in range(dataset["val"].len)]
        
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    
    class_weights = calculate_class_weights([torch.argmax(dataset["train"].get_aspect_label(pos), dim=1).long() for pos in range(dataset["train"].len)])
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # CrossEntropyLoss is suitable for multi-class classification
    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
    num_epochs = 10
    logging.basicConfig(filename='output.log', level=logging.INFO)
    train_model(num_epochs, train_loader, val_loader, criterion, optimizer, num_aspects, model, logging)
    # for epoch in range(num_epochs):
    #     model.train()
    #     for input, attention_mask, label in train_loader:
    #         optimizer.zero_grad() 
    #         # print(input.shape)
    #         # print(attention_mask.shape)
    #         # print(label.shape)
    #         output = model(input, attention_mask)

    #         label = torch.argmax(label, dim=2).long()
    #         output = output.permute(0, 2, 1)
    #         # print(output.shape)
    #         # print(label.shape)
    #         loss = criterion(output, label)
    #         loss.backward()
    #         optimizer.step()
    #         print(epoch)
    #     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
            # print(output.shape)
            # _, predicted = torch.max(output, dim=2)
            # print(predicted)
            # print(predicted.shape)
            

    #train_dataset = [(torch.tensor([token.value for token in entry.text_tokens], dtype=torch.long), entry.text_aspect) for entry in dataset['train']]
    #val_dataset = [(torch.tensor([token.value for token in entry.text_tokens], dtype=torch.long), entry.text_aspect) for entry in dataset['val']]
    #train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    # hidden_dim = 128
    # embedding_dim = 300
    # glove = GloVe(name='6B', dim=embedding_dim)
    # glove_embedding_weights = glove.vectors
    # #model = AspectClassificationModel(vocab_size, embedding_dim, hidden_dim, num_aspects)
    # model = AspectClassificationModel(glove_embedding_weights, hidden_dim, num_aspects)
    # # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # num_epochs = 5
    # train_model(num_epochs, train_loader,val_loader, criterion, optimizer, num_aspects, model)
    
if __name__ == "__main__":
    main()