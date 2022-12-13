from model import SentimentLSTM
from preprocessing import split_word_reviews, dataset_parsing, create_dictionaries, pad_text 

n_vocab = len(word_to_int_dict)
n_embed = 50
n_hidden = 100
n_output = 1
n_layers = 2

net = SentimentLSTM(n_vocab, n_hidden, n_output, n_layers)

