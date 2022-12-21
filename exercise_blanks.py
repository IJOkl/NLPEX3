import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
from matplotlib import pyplot as plt


# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
TEST_NEGATED = "test_negated"
TEST_RARE = "test_rare"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    # vocab = list(wv_from_bin.vocab.keys())
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    cnt = 0
    res = np.zeros(embedding_dim)
    for word in sent.text:
        if word in word_to_vec:
            cnt += 1
            res += word_to_vec[word]
    if cnt == 0:
        return res
    return res / cnt


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_vec = np.zeros(size)
    one_vec[ind] += 1
    return one_vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vec_size = len(word_to_ind)
    res = np.zeros(vec_size)
    for word in sent.text:
        res += get_one_hot(vec_size, word_to_ind[word])
    return res / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return dict(zip(words_list, range(len(words_list))))


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    res = np.zeros((seq_len, embedding_dim))
    all_words = sent.text[:seq_len]
    for i, word in enumerate(all_words):
        if word in word_to_vec:
            res[i] = word_to_vec[word]
    return res


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        neg_idx = data_loader.get_negated_polarity_examples(self.sentences[TEST])
        rare_idx = data_loader.get_rare_words_examples(self.sentences[TEST], self.sentiment_dataset)

        self.sentences[TEST_NEGATED] = [self.sentences[TEST][idx] for idx in neg_idx]
        self.sentences[TEST_RARE] = [self.sentences[TEST][idx] for idx in rare_idx]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                           num_layers=n_layers, bidirectional=True,
                           batch_first=True)
        self.hidden_size = hidden_dim
        self.linear_layer = nn.Linear(2*hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        h_0 = torch.zeros(2, text.size(0), self.hidden_size)
        c_0 = torch.zeros(2, text.size(0), self.hidden_size)
        output, (h_n, c_n) = self.rnn(text, (h_0, c_0))
        # need h_n - the last hidden state - instead the last output (which different in LSTM)
        dropout = self.dropout(torch.cat((h_n[0], h_n[1]), dim=1))
        out = self.linear_layer(dropout)
        return out

    def predict(self, text):
        h_0 = torch.zeros(2, text.size(0), self.hidden_size)
        c_0 = torch.zeros(2, text.size(0), self.hidden_size)
        output, (h_n, c_n) = self.rnn(text, (h_0, c_0))
        out = self.linear_layer(torch.cat((h_n[0], h_n[1]), dim=1))
        output = torch.sigmoid(out)
        output = torch.where(output > 0.5, 1, 0)
        return output


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.layer1 = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        output = self.layer1(x)
        return output

    def predict(self, x):
        output = self.forward(x)
        output = torch.sigmoid(output)
        output = torch.where(output > 0.5, 1, 0)
        return output


# ------------------------- training functions -------------

def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # binary_preds = torch.where(preds > 0.5, 1, 0)
    correct = (preds == y).sum().float()
    return correct / len(preds)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    data_len = len(data_iterator)
    total_loss, total_acc = 0, 0
    preds, lables = torch.empty(0), torch.empty(0)
    for i, (x, y) in enumerate(data_iterator):
        optimizer.zero_grad()
        output = model(x.float())
        cur_pred = model.predict(x.float())
        preds = torch.cat((preds, cur_pred))
        lables = torch.cat((lables, y))
        loss = criterion(torch.squeeze(output), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # if i%100 == 1:
        #     print(f"average loss after {i} steps: {total_loss / i}")
    t_a = binary_accuracy(torch.squeeze(preds), lables)
    return total_loss / data_len, t_a


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    preds, lables = torch.empty(0), torch.empty(0)
    data_len = len(data_iterator)
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for x, y in data_iterator:
            output = model(x.float())
            cur_pred = model.predict(x.float())
            preds = torch.cat((preds, cur_pred))
            lables = torch.cat((lables, y))
            loss = criterion(torch.squeeze(output), y)
            total_loss += loss.item()
    t_a = binary_accuracy(torch.squeeze(preds), lables)
    return total_loss / data_len, t_a


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    preds = []
    for d in data_iter:
        preds.append(model.predict(d))
    return preds


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    print(f"Train {model.__class__.__name__} model")
    t_losses, t_accuracies, v_losses, v_accuracies = [], [], [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t_data_iterator = data_manager.get_torch_iterator(TRAIN)
    v_data_iterator = data_manager.get_torch_iterator(VAL)
    epoch = 0
    model.to(get_available_device())
    criterion.to(get_available_device())
    train_path = f"train_{model.__class__.__name__}.pkl"
    # if os.path.exists(train_path):
    #     model, optimizer, epoch = load(model, "train", optimizer)
    while epoch < n_epochs:
        # print(f"Train epoch {epoch}")
        t_loss, t_acc = train_epoch(model, t_data_iterator, optimizer, criterion)
        # print(f"Evaluate epoch {epoch}")
        v_loss, v_acc = evaluate(model, v_data_iterator, criterion)
        t_losses.append(t_loss)
        t_accuracies.append(t_acc)
        v_losses.append(v_loss)
        v_accuracies.append(v_acc)
        epoch += 1
    save_model(model, train_path, n_epochs, optimizer)
    test_loss, test_acc = evaluate(model, data_manager.get_torch_iterator(TEST), criterion)
    print(f"Test accuracy: {test_acc}\nTest loss: {test_loss}\n")
    _, negated_acc = evaluate(model, data_manager.get_torch_iterator(TEST_NEGATED), criterion)
    _, rare_acc = evaluate(model, data_manager.get_torch_iterator(TEST_RARE), criterion)
    print(f"Test accuracy for negated polarity examples: {negated_acc}")
    print(f"Test accuracy for rare words examples: {rare_acc}\n\n")
    return t_losses, t_accuracies, v_losses, v_accuracies


def plot_graph(t_data, v_data, title, x_label, y_label, file_name):
    fig, ax = plt.subplots()
    ax.plot(t_data, label="Train")
    ax.plot(v_data, label="Validation")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.show()
    fig.savefig(file_name)


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    DM = DataManager(batch_size=64)
    embed_dims = len(DM.sentiment_dataset.get_word_counts())
    log_linear = LogLinear(embed_dims)
    t_losses, t_accuracies, v_losses, v_accuracies = train_model(log_linear, DM, 20, 0.01, weight_decay=0.001)
    plot_graph(t_losses, v_losses, "Loss - one hot", "Epochs", "Loss value", "one_hot_loss.png")
    plot_graph(t_accuracies, v_accuracies, "Accuracy - one hot", "Epochs", "Accuracy value", "one_hot_acc.png")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    DM = DataManager(batch_size=64, data_type=W2V_AVERAGE, embedding_dim=300)
    log_linear = LogLinear(300)
    t_losses, t_accuracies, v_losses, v_accuracies = train_model(log_linear, DM, 20, 0.01, weight_decay=0.001)
    plot_graph(t_losses, v_losses, "Loss - w2v", "Epochs", "Loss value", "w2v_loss.png")
    plot_graph(t_accuracies, v_accuracies, "Accuracy - w2v", "Epochs", "Accuracy value", "w2v_acc.png")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    DM = DataManager(batch_size=64, data_type=W2V_SEQUENCE, embedding_dim=300)
    lstm = LSTM(300, 100, 1, 0.5)
    t_losses, t_accuracies, v_losses, v_accuracies = train_model(lstm, DM, 4, 0.001, weight_decay=0.0001)
    plot_graph(t_losses, v_losses, "Loss - lstm", "Epochs", "Loss value", "lstm_loss.png")
    plot_graph(t_accuracies, v_accuracies, "Accuracy - lstm", "Epochs", "Accuracy value", "lstm_acc.png")


if __name__ == '__main__':
    # device = get_available_device()
    # print(np.ones(3))
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    train_lstm_with_w2v()
