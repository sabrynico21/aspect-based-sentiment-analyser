import logging
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from .utils import read_json_file, calculate_class_weights, get_predicted_labels, get_performances
from .entities import TextWithLabelsContainer
from .constants import aspect_mapping, opinion_mapping
from .model import AspectClassificationModel
torch.manual_seed(42)

if __name__ == "__main__":
    dataset = {}
    for set in ['train', 'val','test']:
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

    model = AspectClassificationModel(bert_model_name='bert-base-uncased', num_aspects=num_aspects, num_opinions=num_opinions)
    for param in model.bert.parameters():
        param.requires_grad = False
    
    train_data = [(dataset["train"].get_tokens(pos),
                  dataset["train"].get_attention_mask(pos),
                  dataset["train"].get_aspect_label(pos),
                  dataset["train"].get_opinion_label(pos)) for pos in range(dataset["train"].len)]
        
    val_data = [(dataset["val"].get_tokens(pos),
                  dataset["val"].get_attention_mask(pos),
                  dataset["val"].get_aspect_label(pos),
                  dataset["val"].get_opinion_label(pos)) for pos in range(dataset["val"].len)]
        
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=True)
    
    aspect_weights = calculate_class_weights([torch.argmax(dataset["train"].get_aspect_label(pos), dim=1).long() for pos in range(dataset["train"].len)])
    opinion_weights = calculate_class_weights([torch.argmax(dataset["train"].get_opinion_label(pos), dim=1).long() for pos in range(dataset["train"].len)])
    
    criterion_aspect = nn.CrossEntropyLoss(weight=aspect_weights)
    criterion_opinion = nn.CrossEntropyLoss(weight=opinion_weights)

    optimizer_aspect = optim.Adam(model.fc1.parameters(), lr=0.001)
    optimizer_opinion = optim.Adam(model.fc2.parameters(), lr=0.001)
    num_epochs = 10
    logging.basicConfig(filename='output.log', level=logging.INFO, filemode='a') 
    #model.train_and_save(num_epochs, train_loader, val_loader, criterion_aspect, criterion_opinion, optimizer_aspect, optimizer_opinion, num_aspects, num_opinions, logging)

    test_data = [(dataset["test"].get_tokens(pos),
                  dataset["test"].get_attention_mask(pos),
                  dataset["test"].get_aspect_label(pos),
                  dataset["test"].get_opinion_label(pos)) for pos in range(dataset["test"].len)]

    test_loader = DataLoader(test_data, batch_size=dataset["test"].len, shuffle=False)
    
    predictions = model.predict(num_aspects, num_opinions, criterion_aspect, criterion_opinion, test_loader)
    
    text_tokens_values = [item[0] for item in test_data]
    texts = dataset["test"].get_texts()
    label_predictions = get_predicted_labels(predictions, tokenizer, text_tokens_values, texts)

    ground_truth = [ dataset["test"].get_text_labels(pos) for pos in range(dataset["test"].len)]  
    get_performances(ground_truth, label_predictions)