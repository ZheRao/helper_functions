import torch



# the unsupervisedTokenizedTextDataset takes in tokenized text (could be at character or word leve, or other level of tokenization), 
# desired block size (input sequence length), predict_size (model output length: default 1 - use input sequence to predict the next token)
# the dataset will have length of len(full_txt) - (block_size+predict_size) +1
# for getitem, when given index
# txt[index], txt[index+1], ... txt[index+block_size - 1] will be input sequence with length block_size
# txt[index+block_size] ,..., txt[index+block_size+predict_size-1] will be output sequence with length predict_size
class unsupervisedTokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self,full_txt,block_size,predict_size=1):
        self.txt = full_txt
        self.block_size = block_size
        self.predict_size = predict_size
    def __len__(self):
        return len(self.txt) - (self.block_size + self.predict_size) + 1
    def __getitem__(self, idx):
        input_sequence = self.txt[idx:idx+self.block_size]
        output_sequence = self.txt[idx+self.block_size:idx+self.block_size+self.predict_size]
        return input_sequence, output_sequence