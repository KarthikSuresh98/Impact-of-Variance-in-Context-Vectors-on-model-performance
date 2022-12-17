from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from visualization import plt_all_attentions, plt_attentions

question = '''What is Machine Learning?'''

paragraph = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                 on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                 decisions without being explicitly programmed to perform the task. '''
           

deberta_tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-squad2")
deberta_model = AutoModelForQuestionAnswering.from_pretrained("deepset/deberta-v3-base-squad2")

encoding = deberta_tokenizer.encode_plus(text=question,text_pair=paragraph)

inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = deberta_tokenizer.convert_ids_to_tokens(inputs) #input tokens

output = deberta_model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]), output_attentions=True, output_hidden_states=True)

start_index = torch.argmax(output.start_logits)
end_index = torch.argmax(output.end_logits)
answer = ' '.join(tokens[start_index:end_index+1])

corrected_answer = ''
for word in answer.split():
  if word[0] == '▁':
    if corrected_answer == '':
      corrected_answer += word[1:]
    else:
      corrected_answer += ' ' + word[1:]
  else:
    corrected_answer += word

print(corrected_answer)

hidden_state = output['hidden_states'][-1]
attentions = output['attentions'][-1][0]
plt_all_attentions(attentions, 'deberta_all_heads.png') 
plt_attentions(attentions[0], tokens, 'deberta_single_head.png')