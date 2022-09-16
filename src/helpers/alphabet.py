import json

class Alphabet:
    def __init__(self, lookup_file):
        with open(lookup_file) as json_file:
            self.label_mapping = json.load(json_file)
            self.reverse_label_mapping = dict((v, k) for k, v in self.label_mapping.items())
        self.size = len(self.label_mapping)
        self.units = []
        for k in list(self.label_mapping.keys()):
            self.units.append(k)




