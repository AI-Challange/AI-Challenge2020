import torch
from torch.utils import data
import os
import json
from sklearn.feature_extraction.text import CountVectorizer


def text_to_seq(context, question, vocab, tokenizer, max_length=128):
    token_question = tokenizer(question)
    token_context = tokenizer(context)

    len_cut_context = max_length - len(token_question) - 1
    len_padding = 0

    if len(token_context) > len_cut_context:
        token_context = token_context[:len_cut_context]
    else:
        len_padding = len_cut_context - len(token_context)

    seq_question = [int(vocab[token]) if token in vocab else 10000000 for token in token_question]
    seq_context = [int(vocab[token]) if token in vocab else 10000000 for token in token_context]

    return [10000002]*len_padding + seq_context + [10000001] + seq_question


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase

        self.data_path = os.path.join(root, self.phase + '.json')
        data_f = open(self.data_path, 'r', encoding='utf-8-sig')
        self.data = json.load(data_f)

        self.vocab_path = os.path.join(root, 'vocab.json')
        vocab_f = open(self.vocab_path, 'r', encoding='utf-8-sig')
        self.vocab = json.load(vocab_f)

        self.tokenizer = CountVectorizer().build_tokenizer()

        self.dataset = [item for topic in self.data['data'] for item in topic['paragraphs']]

        self.processed_data = []
        for d in self.dataset:
            context = d['context']
            qas = d['qas']

            for qa in qas:
                qid = qa['id']
                q = qa['question']
                sequence = text_to_seq(context, q, self.vocab, self.tokenizer, max_length=128)

                if self.phase != 'test':

                    a = qa['answers'][0]['text']
                    a_start = qa['answers'][0]['answer_start']
                    a_end = a_start + len(a) - 1

                    processed_data = {'answer_pos': torch.LongTensor([a_start, a_end]),
                                      'id': qid,
                                      'sequence': torch.LongTensor(sequence),
                                      'context': context}
                else:
                    processed_data = {'id': qid,
                                      'sequence': torch.LongTensor(sequence),
                                      'context': context}

                self.processed_data.append(processed_data)

    def __getitem__(self, index):

        # if self.phase != 'test' :
        return self.processed_data[index]
        # elif self.phase == 'test' :
        #     dummy = ""
        #     return (self.labels['file'][index], image, dummy)

    def __len__(self):
        return len(self.processed_data)

    def get_label_file(self):
        return self.data_path


def data_loader(root, phase='train', batch_size=16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()
