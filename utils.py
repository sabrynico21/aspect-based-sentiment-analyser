import json, torch
import torch.nn.functional as F
import torch.nn as nn
import spacy
from itertools import groupby
from transformers import pipeline
from statistics import mode
from .constants import aspect_mapping, opinion_mapping, stop_words_to_include
from .entities import Label

def calculate_iou_score(ground_truth, predicted):
    ground_truth_words = ground_truth.split()
    predicted_words = predicted.split()
    intersection = sum(1 for word in ground_truth_words if word in predicted_words)
    union = len(ground_truth_words) + len(predicted_words) - intersection
    iou_score = intersection / union if union != 0 else 0
    return iou_score

def find_index_of_opinion(gt_label, pred_labels, list_of_index):
    max_iou_score = 0
    index = "NULL"
    for num, pred_label in enumerate(pred_labels):
        if num not in list_of_index: 
            opinion_iou_score = calculate_iou_score(gt_label.opinion_text, pred_label.opinion_text)
            aspect_iou_score = calculate_iou_score(gt_label.aspect_text, pred_label.aspect_text)
            iou_score = opinion_iou_score + aspect_iou_score
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                index = num
    return index

def get_metrics(gt_label, pred_labels, index):
    iou_score = dict()
    accuracy = dict ()
    if index == "NULL":
        iou_score["text_aspect"] = 0
        iou_score["text_opinion"] = 0
        accuracy["polarity"] = 0
        accuracy["category_aspect"] = 0
        accuracy["category_opinion"] = 0
    else:
        pred_label = pred_labels[index]
        iou_score["text_aspect"]=calculate_iou_score(gt_label.aspect_text, pred_label.aspect_text)
        iou_score["text_opinion"]=calculate_iou_score(gt_label.opinion_text, pred_label.opinion_text)
        accuracy["polarity"] = (1 if gt_label.polarity == pred_label.polarity else 0)
        accuracy["category_aspect"] = (1 if gt_label.category_aspect == pred_label.category_aspect else 0)
        accuracy["category_opinion"] = (1 if gt_label.category_opinion == pred_label.category_opinion else 0)
    return iou_score, accuracy

def get_performances(ground_truth, predicted):
    iou_score = dict()
    accuracy = dict ()
    mean_iou_score = dict()
    mean_accuracy = dict()
    mean_iou_score["text_aspect"] = 0
    mean_iou_score["text_opinion"] = 0
    mean_accuracy["polarity"] = 0
    mean_accuracy["category_aspect"] = 0
    mean_accuracy["category_opinion"] = 0
    mean_recall = AverageValueMeter()
    mean_precision = AverageValueMeter()
    for gt_labels, pred_labels in zip(ground_truth, predicted):
        iou_score["text_aspect"] = 0
        iou_score["text_opinion"] = 0
        accuracy["polarity"] = 0
        accuracy["category_aspect"] = 0
        accuracy["category_opinion"] = 0
        list_of_index = []
        recall = 0
        precision = 0 
        for gt_label in gt_labels:
            index = find_index_of_opinion(gt_label,pred_labels, list_of_index) 
            list_of_index.append(index)
            metrics = get_metrics(gt_label, pred_labels, index)

            iou_score["text_aspect"] += metrics[0]["text_aspect"]
            iou_score["text_opinion"] += metrics[0]["text_opinion"]
            accuracy["polarity"]+= metrics[1]["polarity"]
            accuracy["category_aspect"]+=metrics[1]["category_aspect"]
            accuracy["category_opinion"]+=metrics[1]["category_opinion"]
        precision = (len(list_of_index) - list_of_index.count("NULL")) / len (pred_labels)
        recall = (len(list_of_index) - list_of_index.count("NULL")) / len(list_of_index)
        mean_recall.add(recall, 1)
        mean_precision.add(precision,1)
         
        mean_iou_score["text_aspect"] += iou_score["text_aspect"] / len(gt_labels)
        mean_iou_score["text_opinion"] += iou_score["text_opinion"] / len(gt_labels)
        mean_accuracy["polarity"] += accuracy["polarity"] / len(gt_labels)
        mean_accuracy["category_aspect"] += accuracy["category_aspect"] / len(gt_labels)
        mean_accuracy["category_opinion"] += accuracy["category_opinion"] / len(gt_labels)
   
    print("Recall: " , mean_recall.compute())
    print("IoU score - Aspect: ", mean_iou_score["text_aspect"] / len(predicted))
    print("IoU score - Opinion: ", mean_iou_score["text_opinion"] / len(predicted))
    print("Accuracy - Polarity: ", mean_accuracy["polarity"] / len(predicted))
    print("Accuracy - Aspect category: ", mean_accuracy["category_aspect"] / len(predicted))
    print("Accuracy - Opinion category: ", mean_accuracy["category_opinion"] / len(predicted))
    print("Precision: ", mean_precision.compute())

def remove_stopwords(doc, type ):
    if type == "aspect":
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.text.endswith(("ly", "est"))]
    elif( (len(doc) == 1 and doc[0].is_stop and doc[0].text not in stop_words_to_include)):
        filtered_tokens = []
    else: 
        filtered_tokens = [token.text for token in doc ]
    return (" " if filtered_tokens == [] else ' '.join(filtered_tokens))

def compute_value(tokens, positions):
    result = []
    for pos in positions:
        result.append(tokens[pos])
    return result

def opinion_relative_text_exists(parsing, text_opinion, start_index, len_opinion):
    formed_value = ' '.join(token.text.lower() for token in parsing[start_index:start_index + len_opinion])
    return formed_value == text_opinion

def get_entity_and_labels_predictions(index, predictions, tokenizer, token_values):
    entity_prediction = { } 
    labels_prediction = { }
    tokens = token_values[index]
    for type in ["aspect","opinion"]:  
        entity_prediction[type] = []
        labels_prediction[type] = []
        pred = predictions[type][index]
        grouped = [(key, [x[0] for x in group]) for key, group in groupby(enumerate(pred), key=lambda x: x[1] != 0)]
        labels_prediction[type] = [pred[positions] for key, positions in grouped if key]
        tokens_sequences = [compute_value(tokens, positions) for key, positions in grouped if key] 
        entity_prediction[type] = [tokenizer.decode(sequence) for sequence in tokens_sequences]
    return entity_prediction, labels_prediction

def get_attr(parsing, text_opinion):
    start_index = None 
    len_opinion = len(text_opinion.split()) 
    for num, split in enumerate(text_opinion.split()):
        start_index = next((index for index, token in enumerate(parsing) if token.text.lower().startswith(split)), None)
        if start_index is not None:
            len_opinion = len_opinion - num
            text_opinion = text_opinion[num:]
            break
    return start_index, text_opinion, len_opinion

def map_aspect_to_opinion(nlp, entity_prediction, label_info, parsing, current_node):
    found = False
    while not found:
        for num_as, text_aspect in enumerate(entity_prediction["aspect"]):
            if current_node.head.text.lower() in text_aspect:
                label_info["aspect"] = remove_stopwords(nlp(text_aspect), "aspect")
                label_info["aspect_number"] = num_as
                found = True
                return found 
        if (current_node.text.lower() == current_node.head.text.lower()):
            break
        current_node = current_node.head

    if( current_node.text.lower() == current_node.head.text.lower()):
        for children in current_node.head.children:
            for num_as, text_aspect in enumerate(entity_prediction["aspect"]):
                if (children.text.lower() in text_aspect):
                    found = True 
                    label_info["aspect"] = remove_stopwords(nlp(text_aspect), "aspect")
                    label_info["aspect_number"] = num_as
                    return found 
    return found

def connect_opinion_to_aspect(index, parsing, nlp, entity_prediction, labels_prediction, label_info ):
    current_node = parsing[index]           
    found_aspect_for_opinion = map_aspect_to_opinion(nlp, entity_prediction, label_info, parsing, current_node)
            
    if found_aspect_for_opinion:
        num_aspect = mode(labels_prediction["aspect"][label_info["aspect_number"]]) 
        label_info["aspect_category"] = int(num_aspect.item())
    else:
        label_info["aspect"] = "NULL"
        label_info["aspect_category"] = aspect_mapping["RESTAURANT"] 

def find_entity(entity, text, num, nlp, parsing, labels_prediction, sentiment_analyzer):
    #for num_op, text_opinion in enumerate(entity_prediction[entity]): 
    label_info = dict()
    text = remove_stopwords(nlp(text), entity) 
    start_index, text, len_entity = get_attr(parsing, text)         
    if start_index is None:
        #print(f"Could not find the starting token for '{text_opinion}'.")
        return None, label_info
    if not opinion_relative_text_exists(parsing, text, start_index, len_entity):
        #print(f"The subsequent tokens do not form exactly the value '{text_opinion}'.")
        return None, label_info
    
    if len(text) < 3:
        return None, label_info

    label_info[entity] = text
    num_entity = mode(labels_prediction[entity][num]) 
    label_info[f"{entity}_category"] = int(num_entity.item())
    label_info["polarity"] = sentiment_analyzer(text)[0]["label"].lower()

    return start_index + len(text.split())-1, label_info
        

def get_predicted_labels(predictions, tokenizer, token_values, texts ):
    nlp = spacy.load("en_core_web_sm")
    labels = [ ] 
    sentiment_analyzer = pipeline("sentiment-analysis")

    for i in range(predictions["aspect"].shape[0]):
        entity_prediction, labels_prediction = get_entity_and_labels_predictions(i, predictions, tokenizer, token_values)
        text = texts[i]
        labels_dicts = [ ]
        parsing = nlp(text)
        sentence_aspects = [ ]

        for num_op, text_opinion in enumerate(entity_prediction["opinion"]): 
            index_of_opinion, label_info = find_entity("opinion", text_opinion,num_op, nlp, parsing, labels_prediction, sentiment_analyzer)
            if (index_of_opinion == None ):
                continue
            connect_opinion_to_aspect(index_of_opinion, parsing, nlp, entity_prediction, labels_prediction, label_info)

            if (label_info["aspect"] != label_info["opinion"]):          
                labels_dicts.append(Label(aspect_text = label_info["aspect"], opinion_text = label_info["opinion"], polarity = label_info["polarity"], category1=label_info["aspect_category"], category2=label_info["opinion_category"]))
                sentence_aspects.append(label_info["aspect"])
        
        for num_as, text_aspect in enumerate(entity_prediction["aspect"]):
            text_aspect = remove_stopwords(nlp(text_aspect), "aspect")
            if (text_aspect not in sentence_aspects):
                index_of_aspect, label_info = find_entity("aspect", text_aspect,num_as, nlp, parsing, labels_prediction, sentiment_analyzer)
                if (index_of_aspect == None ):
                    continue
                labels_dicts.append(Label(aspect_text = label_info["aspect"], opinion_text = "NULL", polarity = sentiment_analyzer(text)[0]["label"].lower(), category1= label_info["aspect_category"], category2= opinion_mapping["GENERAL"]))
        
        if labels_dicts == []:
            labels_dicts.append(Label(aspect_text = "NULL", opinion_text = "NULL", polarity = sentiment_analyzer(text)[0]["label"].lower(), category1= aspect_mapping["RESTAURANT"], category2= opinion_mapping["MISCELLANEOUS"]))
        labels.append(labels_dicts)  
        #print( [(label.aspect_text, label.opinion_text) for label in labels[i]])  
    return labels 

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

def read_json_file(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data_list

def calculate_class_weights(labels):
    concatenated_labels = torch.cat(labels, dim=0)
    _, class_counts = torch.unique(concatenated_labels, return_counts=True)
    class_frequencies = class_counts.float() / class_counts.sum().float()
    class_weights = 1.0 / class_frequencies
    class_weights /= class_weights.sum()
    return class_weights