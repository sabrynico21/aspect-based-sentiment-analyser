import torch
from .constants import aspect_mapping, opinion_mapping
#from .utils import compare_values

def compare_values(target, tokens):
    for token in tokens:   
        if target.value == token:
            return "True"
    return "False"

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
    
class Label:
    def __init__(self, aspect_text=None, opinion_text=None, aspect_pos=None, opinion_pos=None, polarity=None, category1=None, category2=None):
        self._aspect_text = aspect_text
        self._opinion_text = opinion_text 
        self._aspect_pos = aspect_pos
        self._opinion_pos = opinion_pos 
        self._polarity = polarity 
        self._category_aspect = category1
        self._category_opinion = category2

    @property
    def aspect_text(self):
        return self._aspect_text
    
    @property
    def opinion_text(self):
        return self._opinion_text
    
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
    def __init__(self, text, tokens=None, labels=None):
        self._text = str(text)
        self._text_tokens = tokens if tokens is not None else []
        self._text_labels = labels if labels is not None else []
    
    @property
    def text(self):
        return self._text
    
    @property
    def text_tokens(self):
        return self._text_tokens
    
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
        text_aspect = [aspect_mapping["NULL"]] * len(self.text_tokens)
        text_opinion = [opinion_mapping["NULL"]] * len(self.text_tokens)
        for label in self.text_labels:
            for pos in label.aspect_pos:
                text_aspect[pos] = label.category_aspect
            for pos in label.opinion_pos:
                text_opinion[pos] = label.category_opinion
        text_aspect=torch.eye(num_aspects)[text_aspect]
        text_opinion=torch.eye(num_opinions)[text_opinion]
        return text_aspect, text_opinion
    
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
         return torch.tensor([token.value for token in self._instances[pos].text_tokens], dtype=torch.long)
    def set_text_tokens(self, sentence, pos, tokenizer):
        tokens = [Token(text = tokenizer.decode(token, skip_special_tokens=True), 
                        value=token.item(),
                        position=position) 
                        for position, token in enumerate(sentence)]
        self._instances[pos].text_tokens = tokens 

    def set_text_labels(self, labels, pos, tokenizer):
        for entry in labels:
            aspect_pos = []
            aspect_value = []
            opinion_pos = []
            opinion_value =[]
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
                    aspect_value.append(token.value)
                if compare_values(token, tokens_opinion) == "True":
                    opinion_pos.append(token.position)
                    opinion_value.append(token.value)
            self._instances[pos].text_labels = Label(aspect_text = (tokenizer.decode(aspect_value) if aspect_value != [] else "NULL"), 
                                                     opinion_text = (tokenizer.decode(opinion_value) if opinion_value != [] else "NULL"), 
                                                     aspect_pos = aspect_pos, 
                                                     opinion_pos = opinion_pos, 
                                                     polarity = sentiment, 
                                                     category1=aspect_mapping[category[0]], 
                                                     category2=opinion_mapping[category[1]])

    def compute_one_hot_format(self, pos, num_aspects, num_opinions):
        aspect, opinion = self._instances[pos].compute_one_hot_format(num_aspects, num_opinions)
        self._aspect_label.append(aspect)
        self._opinion_label.append(opinion)

    def set_attention_mask(self, attention_mask):
        self._attention_mask.append(attention_mask)
    
    def get_aspect_label(self, pos):
        return self._aspect_label[pos]
        
    def get_opinion_label(self, pos):
        return self._opinion_label[pos]

    def get_attention_mask(self, pos):
        return self._attention_mask[pos]
    
    def get_text_labels(self, pos):
        return self._instances[pos].text_labels