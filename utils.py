import json, torch

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

def compare_values(target, tokens):
    for token in tokens:   
        if target.value == token:
            return "True"
    return "False"

def calculate_class_weights(labels):
    concatenated_labels = torch.cat(labels, dim=0)
    _, class_counts = torch.unique(concatenated_labels, return_counts=True)
    class_frequencies = class_counts.float() / class_counts.sum().float()
    class_weights = 1.0 / class_frequencies
    class_weights /= class_weights.sum()
    return class_weights