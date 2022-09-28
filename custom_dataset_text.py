import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to set up a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    #I love Peanuts=['i', 'love', 'peanuts']
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    #adding all captions to list
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        ##length
        def __len__(self):
            return len(self.df)

        #Load the image
        def __getitem__(self, index):
            caption = self.captions[index]
            img_id = self.imgs[index]
            img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

            #transformation of image
            if self.transform is not None:
                img = self.transform(img)

            #Convert vocab(string) to numerical value
            numericalized_caption = [self.vocab.stoi["<SOS>"]] #stoi=string to integer, SOS=Start of Sentence
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption.append(self.vocab.stoi["<EOS>"]) #EOS=End of Sentence

            return img, torch.tensor(numericalized_caption)

