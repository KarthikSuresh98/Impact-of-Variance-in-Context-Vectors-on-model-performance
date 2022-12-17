## Impact of Variance in Context Vectors on model performance

This is the official implementation of the prject for the Special Topics in Artificial Intelligence - Applications of Deep Learning course. Attention mechanism has revolutionized the field of deep learning and attention-based models have achieved great success in many artificial intelligence fields, such as natural language processing, computer vision, and audio processing. As the models become increasingly complex and learn decision-making functions from data, interpreting why a model shows superior performance has attracted lots of interest from academic and industry researchers. In this paper, we hope to examine the attention mechanism, specifically the context vectors. The question we want to ask is: if an attention-based model achieves better performance than another model, does that translate to the former model having more discriminative context vectors? Observing any such patterns will lead to gaining insights into how attention-based models learn to perform better and can help us in designing better architectures to advance the field forward. Experimental results indicate a marginally positive relationship between the two variables and further analyses and visualization of attention weights hint at an observable pattern between learned attention
weights and model performance.

#### Environment setup
The requirements.txt file has the necessary packages needed to reproduce the results
```bash
pip install -r requirements.txt
```

#### Instructions
1. To reproduce the scatterplot results and the calculate the pearson correlation coefficient run:

For Question Answering Task-
```bash
python qa_task.py
```

For Sequence Classification Task-
```bash
python seq_task.py
```


2. attention_heads.py is an example script for visualizing the attention heads. This script does it for Deberta-base
```bash
python attention_heads.py
```

#### References
1. https://huggingface.co/docs/transformers/v4.17.0/en/tasks/question_answering
2. https://huggingface.co/docs/transformers/tasks/sequence_classification


