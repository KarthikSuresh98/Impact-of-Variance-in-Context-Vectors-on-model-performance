from random import shuffle
import random
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
from datasets import load_dataset
from evaluate import load
import statistics

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from visualization import scatter_plot


imdb = load_dataset('imdb')
random.seed(1)
test_set = list(imdb['test'])
shuffle(test_set)

f1_metric = load("f1")

def evaluate(checkpoint = "pig4431/IMDB_BERT_5E", model_name = "Bert-base"):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    results = []

    for i in [0, 2000, 5000, 10000]:
        sim_metric = []
        variance_metric = []

        num_samples = 0
        sim_metric = []
        variance_metric = []
        predictions = []
        references = []

        for j in range(i, len(test_set)):
            if num_samples == 500:
                break

            text = test_set[j]['text']
            encoding = tokenizer.encode_plus(text=text)
            inputs = encoding['input_ids']  #Token embeddings
            sentence_embedding = encoding['token_type_ids']  #Segment embeddings

            if len(inputs) > 512:
                continue
            
            else:
                num_samples = num_samples + 1
                output = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]), output_attentions=True, output_hidden_states=True)

                label = torch.argmax(F.softmax(output.logits, dim=1)).item()
                predictions.append(label)
                references.append(test_set[j]['label'])

                hidden_state = output['hidden_states'][-1]
                attentions = output['attentions'][-1]
                seq_length = attentions.shape[2]

                hidden_state = hidden_state.squeeze(0).detach().numpy()

                all_sim = np.ones((seq_length, seq_length))
                cosine_sim = 1 - cdist(hidden_state, hidden_state, metric='cosine')
                metric = ((all_sim - cosine_sim) ** 2).mean()
                sim_metric.append(metric)

                c_mean = np.mean(hidden_state, axis = 1, keepdims = True)
                c_std = np.std(hidden_state, axis = 1, keepdims = True)
                normalized_context_vector = (hidden_state - c_mean)/c_std

                variance = np.var(normalized_context_vector, axis = 0, keepdims = True)
                variance = np.mean(variance)
                variance_metric.append(variance)
        
        f1_results = f1_metric.compute(predictions=predictions, references=references)      
        results.append({'F1 Score': f1_results['f1'], 'Variance': statistics.mean(variance_metric), 'CSE': statistics.mean(sim_metric), 'Model': model_name})
    
if __name__ == "__main__":
    results = evaluate(checkpoint = "pig4431/IMDB_BERT_5E", model_name = "Bert-base") + evaluate(checkpoint = "pig4431/IMDB_ALBERT_5E", model_name = "Albert-base") + evaluate(checkpoint = "pig4431/IMDB_ELECTRA_5E", model_name = "Electra-base") + evaluate(checkpoint = "pig4431/IMDB_XLNET_5E", model_name = "Xlnet-base") 
    df = pd.DataFrame(results)
    scatter_plot(df, 'seq', eval_metric='var')
    scatter_plot(df, 'seq', eval_metric='cse')