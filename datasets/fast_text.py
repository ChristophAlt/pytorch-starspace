import os
import csv

from torchtext import data


class FastText(data.Dataset):
    """
    Fast text format
    """

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, label_prefix, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        with open(os.path.expanduser(path), 'r') as tsv_file:
            for line in csv.reader(tsv_file, delimiter='\t'):
                lhs_tokens = []
                rhs_tokens = []

                for token in line:
                    if token.startswith(label_prefix):
                        rhs_tokens.append(token)
                    else:
                        lhs_tokens.append(token)

                examples.append(data.Example.fromlist([" ".join(lhs_tokens), " ".join(rhs_tokens)], fields))

        super().__init__(examples, fields, **kwargs)
