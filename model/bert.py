
import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import dense_to_sparse
from supar import Parser
import stanza
import json
from chemdataextractor import Document
import requests
import os
import re



def split_text(text, max_length,tokenizer):
    """根据BERT的最大输入长度将文本分割成块"""
    text=text.strip()
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    for sentence in sentences:
        
        if len(tokenizer.tokenize('. '.join(current_chunk + [sentence]))) <= max_length:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append('. '.join(current_chunk)+".")
                current_chunk = [sentence]
    
    # 添加最后一块
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    #print(len(chunks))
    return chunks


# text batch level
def parse_iupac_batch(texts):
    PLACE_HOLDER = 'COMPOUND'
    processed_texts = []
    iupac_lists = []

    for text in texts:
        doc = Document(text)
        
        # Extract potential IUPAC names
        #iupac_names = [entity["names"][0] for entity in doc.records.serialize() if '-' in entity['names'][0]]
        iupac_names =[entity.text for entity in doc.cems if '-' in entity.text or '(' in entity.text]
        iupac_names.sort(key=len, reverse=True)
        
        # Replace IUPAC names in the text with numbered placeholders
        for index, iupac in enumerate(iupac_names):
            #placeholder = f"{PLACE_HOLDER}{index}"
            placeholder =f"{PLACE_HOLDER}{str(index).zfill(2)}"
            text = text.replace(iupac, placeholder)
        
        iupac_list = []  # Store the IUPAC names corresponding to each COMPOUND
        match = re.findall(r'COMPOUND\d{2}', text)

        if match:
            for cmpd in match:
                iupac_list.append(iupac_names[int(cmpd[len(PLACE_HOLDER):])])
                #print(cmpd,iupac_names[int(cmpd[len(PLACE_HOLDER):])])
        text = re.sub(r'COMPOUND\d{2}', 'COMPOUND', text)
        processed_texts.append(text)
        iupac_lists.append(iupac_list)
    #print("replaced text:",processed_texts)
    return processed_texts, iupac_lists

def text2parsetree_batch(text):
    # 将text转换为解析树
    replaced_text, iupac_list = parse_iupac_batch(text)
    torch.cuda.set_device('cuda:0')  
    parser = Parser.load("/public/home/weixy2024/momunew/model/ptb.biaffine.dep.lstm.char")
    dataset = parser.predict(replaced_text, lang='en', prob=True, verbose=False)
    #print(dataset[0])
    #print(f"arcs:  {dataset.arcs[0]}\n"
          #f"rels:  {dataset.rels[0]}\n")
    return list(dataset.words), dataset.arcs,  dataset.rels,  iupac_list, replaced_text    
    


class textGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(textGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
       
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('/public/home/weixy2024/momunew/bert_pretrained_uncased/')
        self.bertmodel = BertModel.from_pretrained('/public/home/weixy2024/momunew/bert_pretrained_uncased/')
        self.textGNN = textGNN(768, 768, 768)  # Adjust hidden dimensions as needed
        self.dropout = nn.Dropout(0.1)

    def forward(self, texts):
        max_length = 512
        #texts=fix_encoding_batch(texts)  # 处理乱码
        # Preprocess texts
        texts = [re.sub(r',\s+', ',', text) for text in texts]
        #print("batch texts:",texts)
        
        tokens_batch, arcs_batch, rels_batch, iupac_lists_batch, replaced_texts_batch = text2parsetree_batch(texts)
        
        outputs_batch=[]
        for text, tokens, arcs, rels, iupac_list, replaced_text in zip(texts, tokens_batch, arcs_batch,  rels_batch, iupac_lists_batch, replaced_texts_batch):
            #print("text:",text)
            #print("tokens:",tokens)
            #print("replaced_text:",replaced_text)
            root_id=rels.index("root")  # 根节点的id
            chunk=split_text(text, max_length,self.tokenizer)
            #print("chunk_len:",len(chunk))
            #print("chunk:",chunk)
            tokenized_text_chunks=[]
            last_hidden_states_chunks=[]
            for text in chunk:
                #print()
                inputs = self.tokenizer(text=text,
                                   truncation=True,
                                   #padding='max_length',
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   return_tensors='pt').to('cuda')
                tokenized_text=self.tokenizer.tokenize(text)
                tokenized_text_chunks.extend(tokenized_text)
                #print(self.tokenizer.tokenize(text))
                #print(inputs["input_ids"].shape)
                #print("inputs['input_ids']:",inputs["input_ids"].shape)
                with torch.no_grad():
                    outputs = self.bertmodel(**inputs) 
                last_hidden_states = outputs.last_hidden_state  # 获取最后一个隐藏层的输出
            #     maxlen=last_hidden_states.shape[1]
                #print("last_hidden_states:",last_hidden_states.shape)
                last_hidden_states_chunks.append(last_hidden_states)
            #print(len(tokenized_text_chunks))
            last_hidden_states_chunks=torch.cat(last_hidden_states_chunks, dim=1)
            word_emb_list=[]  # 储存parse tree中每个word的向量表示

            k=0 # 遍历hidden state的索引
            compound_idx=0  # 遍历所有COMPOUND
            flagmaxlen=0
            parse_token=tokens
            
            for w in parse_token:  # 遍历解析树的每个节点word
                #print("w:",w)
                
                # 如果是compound，（由于compund表示的iupac中可能含有空格，所以必须先处理compound）
                if w=='COMPOUND':  # 如果当前节点的word是COMPOUND，则将其所对应的iupac单独进行编码作为COMPOUND的表示
                    #print(w)
                    temp_list=[]
                    iupac = iupac_list[compound_idx]
                    #print("iupac:",iupac)
                    subws=self.tokenizer.tokenize(iupac)  # 对iupac分词
                    #print("subws:",subws)
                    temp_list=[]  # 当前iupac对应的所有token的embedding
                    for subw in subws:
                        temp_list.append(last_hidden_states_chunks[0][k])
                        #print("tokenized_text:",tokenized_text_chunks[k])
                        if subw == tokenized_text_chunks[k]:  
                            k+=1
            
                    iupac_rep = temp_list[0].tolist()  # 取第一个token的嵌入向量作为整个word的表示，或者取各个分量的max值
                    word_emb_list.append(iupac_rep)
                    compound_idx+=1
                    
            
                # 如果不是compound
                else:
                    subws=self.tokenizer.tokenize(w)  # 对每个word分词
                    #print("subws:",subws)
                    #print("subws:",subws)
                    temp_list=[]  # 当前word分词后的所有token的embedding
                    for subw in subws:
                        temp_list.append(last_hidden_states_chunks[0][k])
                        #print("tokenized_text:",tokenized_text_chunks[k])
                        if subw == tokenized_text_chunks[k]:  
                            k+=1
                    
                    word_emb_list.append(temp_list[0].tolist())
                #print()
            
            
            
            word_emb_list=torch.tensor(word_emb_list)
            
            # 获得edge_index
            num_tokens = len(tokens)
            adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
            for i, head in enumerate(arcs):
                adj_matrix[head-1, i] = 1  # 将解析树看作有向图
            adj_matrix=torch.tensor(adj_matrix)
            edge_index, _ = dense_to_sparse(adj_matrix)
            
            
            
            # 获得textGNN的输出
            data=Data(x=word_emb_list, edge_index=edge_index).to('cuda')
            out = self.textGNN(data.x,data.edge_index)
            out = self.dropout(out)
            #print("out[root_id]:",out[root_id].shape)
            outputs_batch.append(out[root_id].unsqueeze(0))

        outputs_batch=torch.cat(outputs_batch, dim=0)
        #print("outputs_batch.shape:",outputs_batch.shape)
        return outputs_batch
if __name__ == '__main__':
    model = TextEncoder().to('cuda')
    text=['N4-Hydroxyctidine,or EIDD-1931,is a ribonucleoside analog which induces mutations in RNA virions. it was first described in the literature in 1980 as a potent mutagen of bacteria and phage. It has shown antiviral activity against Venezuelan equine encephalitis virus,and the human coronavirus HCoV-NL63 in vitro. N4-hydroxycytodine has been shown to inhibit SARS-CoV-2 as well as other human and bat coronaviruses in mice and human airway epithelial cells. It is orally bioavailable in mice and distributes into tissue before becoming the active 5â\x80\x99-triphosphate form,which is incorporated into the genome of new virions,resulting in the accumulation of inactivating mutations. In non-human primates,it was poorly orally bioavailable. A [remdesivir] resistant mutant mouse hepatitis virus has also been shown to have increased sensitivity to it. The prodrug of it,[EIDD-2801],is also being investigated for its broad spectrum activity against the coronavirus family of viruses.']
    
    #model.load_state_dict(torch.load('test_weights.pth'))
    model.eval()
    model(text)
    
    
