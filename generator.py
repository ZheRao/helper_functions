import os
import torch
class generator:
    def __init__(self, model, encoder, decoder, model_name):
        self.model = model.cpu()
        self.encoder = encoder
        self.decoder = decoder
        self.model_name = model_name
        self.folder_name = "generated_text"
    
    def _open_file(self):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.f = open(os.path.join(self.folder_name,self.model_name)+".txt","a")

    
    def generate_without_prompt(self, user_input, generation_length=300, block_size=512,default_start_token=True,pad_with=32):
        self._open_file()
        output_list = [user_input]
        if default_start_token: user_input = '<|startofchapter|>' + user_input
        # tokenize user input
        input_tokens = torch.tensor(self.encoder(user_input),dtype=torch.long)
        # pad tokens
        if len(input_tokens) < block_size:
            tokens = torch.full(size=(1,block_size),fill_value=pad_with,dtype=torch.long)
            tokens[0,-len(input_tokens):] = input_tokens
        else:
            tokens = input_tokens[-block_size:].unsqueeze(0)
        
        m = f"User input: {user_input}\nGenerating----------------------------------------------------\n"
        print(m)
        self.f.write("\n\n"+m+"\n")

        print_status = False
        print_idx_start = 0
        for i in range(generation_length):
            if i % 30 == 1:
                print_status = True
            if ("." in output_list[-1] or "," in output_list[-1]) and print_status==True:
                output_sequence = "".join(output_list[print_idx_start:])
                print_idx_start = len(output_list)
                print_status=False
                print(output_sequence)
                self.f.write(output_sequence+"\n")
            tokens_truncate = tokens[0,-block_size:]
            logit = self.model(tokens_truncate)
            logit = logit[0,-1,:]
            prob = torch.nn.functional.softmax(logit,dim=0)
            new_token = torch.multinomial(input=prob,num_samples=1)
            tokens = torch.cat((tokens,new_token.unsqueeze(0)),dim=1)
            output_list.append(self.decoder([new_token.item()]))
        m = f"\nEnd of generation------------------------------------------------------------------\n"
        print(m)
        self.f.write(m+"\n\n")
        self.f.close()
        return output_list
            


        