from random import shuffle
import random
import torch
from scipy.spatial.distance import cdist
import numpy as np
from datasets import load_dataset
from evaluate import load
import statistics

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import pandas as pd
from visualization import scatter_plot


squad_metric = load("squad_v2")
squad = load_dataset("squad_v2")
random.seed(1)
val = list(squad['validation'])
shuffle(val)


def evaluate(checkpoint = "twmkn9/bert-base-uncased-squad2", model_name = "Bert-base"):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    results = []

    for i in [0, 1000, 2000, 5000]:
        sim_metric = []
        variance_metric = []

        num_samples = 0
        sim_metric = []
        variance_metric = []
        predictions = []
        references = []

        for j in range(i, len(val)):
            if num_samples == 200:
                break

            context = val[j]['context']
            question = val[j]['question']
            encoding = tokenizer.encode_plus(text=question,text_pair=context)
            inputs = encoding['input_ids']  #Token embeddings
            sentence_embedding = encoding['token_type_ids']  #Segment embeddings

            if len(inputs) > 384 or len(val[j]['answers']['text']) == 0:
                continue
            
            else:
                num_samples = num_samples + 1
                tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
                output = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]), output_attentions=True, output_hidden_states=True)

                start_index = torch.argmax(output.start_logits)
                end_index = torch.argmax(output.end_logits)
                answer = ' '.join(tokens[start_index:end_index+1])

                corrected_answer = ''
                
                if model_name == 'Electra-base' or model_name == 'Bert-base':
                    for word in answer.split():
                        #If it's a subword token
                        if word[0:2] == '##':
                            corrected_answer += word[2:]
                        else:
                            corrected_answer += ' ' + word
   
                elif model_name == 'Albert-base':
                    for word in answer.split():
                        if word[0] == '▁':
                            if corrected_answer == '':
                                corrected_answer += word[1:]
                            else:
                                corrected_answer += ' ' + word[1:]
                        else:
                            corrected_answer += word
 
                elif model_name == 'Deberta-base':
                    for word in answer.split():
                        if word[0] == '▁':
                            if corrected_answer == '':
                                corrected_answer += word[1:]
                            else:
                                corrected_answer += ' ' + word[1:]
                        else:
                            corrected_answer += word
                
                else:
                    for word in answer.split():
                        if word[0] == 'Ġ':
                            if corrected_answer == '':
                                corrected_answer += word[1:]
                            else:
                                corrected_answer += ' ' + word[1:]
                        else:
                            corrected_answer += word


                references.append({'answers' : val[j]['answers'] , 'id' : val[j]['id']})
                if corrected_answer == '':
                    no_answer_probability = 1
                else:
                    no_answer_probability = 0
                predictions.append({'prediction_text' : corrected_answer, 'id' : val[j]['id'] , 'no_answer_probability' : no_answer_probability})

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
        
        metrics = squad_metric.compute(predictions=predictions, references=references)
        results.append({'F1 Score': metrics['f1'], 'Variance': statistics.mean(variance_metric), 'CSE': statistics.mean(sim_metric), 'Model': model_name})
    
if __name__ == "__main__":
    results = evaluate(checkpoint = "twmkn9/bert-base-uncased-squad2", model_name = "Bert-base") + evaluate(checkpoint = "deepset/roberta-base-squad2", model_name = "Roberta-base") +  evaluate(checkpoint = "deepset/deberta-v3-base-squad2", model_name = "Deberta-base") + evaluate(checkpoint = "deepset/electra-base-squad2", model_name = "Electra-base") + evaluate(checkpoint = "twmkn9/albert-base-v2-squad2", model_name = "Albert-base") 
    df = pd.DataFrame(results)
    scatter_plot(df, 'qa', eval_metric='var')
    scatter_plot(df, 'qa', eval_metric='cse')

    print('Pearson Correlation Coefficient:')
    var = df['Variance']
    cse = df['CSE']
    F1 = df['F1 Score']

    print(np.corrcoef(F1, var))
    print(np.corrcoef(F1, cse))