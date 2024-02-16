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