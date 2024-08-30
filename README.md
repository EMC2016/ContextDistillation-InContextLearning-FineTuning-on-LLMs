# ContextDistillation-InContextLearning-FineTuning-on-LLMs

## Background


This project aims to tackle the challenge of in-context learning (ICL) within Large Language Models (LLMs). [1,2] These models traditionally demand significant memory due to long context handling. While existing solutions advocate for training models from scratch with extended context length, this project proposes exploring alternative fine-tuning methods to alleviate the context length constraint without necessitating expensive re-training or large dataset fine-tuning. 

The initiative particularly focused on employing the "Context Distillation" method for fine-tuning in the NLI classification task, contrasting it against ICL approach as outlined in previous works. [3] This enhanced LLM performances in handling long dialogue conversations, answering queries on large documents, or aiding in code auto-completion with extensive repository knowledge, without the encumbrance of substantial memory requirements. 


## Approach

In context distillation, the prompt is divided into two segments. [3-5] The first part, denoted as C, consists of several human-annotated pairs of questions and corresponding answers. The second part, denoted as X, is a specific question which is answered by the LLM. Initially, the whole prompt is fed into the LLM. The log probabilities, p(X|C), representing classification outcomes, are preserved as the correct outputs. Sequentially, the LLM processes only the second part of the prompt and produces the log probabilities p(X).  To achieve fine-tuning, the LLM employs gradient descent with a loss function based on Kullback–Leibler (KL) divergence between p(X|C) and p(X). [6] The divergence effectively measures the difference between the two probability distributions. The goal, with regards to fine-tuning, is to minimize loss. 

We will implement context distillation-based method to fine-tuning a range of LLMs, each with varying parameters. We’ll use llmft [7] codebase as a starting point and will utilize multiple datasets for the training phase, provided below. Performance and computation cost will be compared with the ICL approach with regards to all datasets and LLMs. In-domain and out-of-domain accuracies will be calculated to describe performance. CPU time (User, System and Total) and Memory Usage (Peak and Average) will be recorded to measure the computation cost in the training process. More methods, such as Vanilla fine tuning with LoRa and Pattern-based fine-tuning with LoRa, could also be implemented to optimize the fine-tuning process if necessary. 







## Reference

[1]: “Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation”, M.Mosbach et al. https://aclanthology.org/2023.findings-acl.779.pdf

[2] “Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning”, Haokun Liu et.https://arxiv.org/abs/2205.05638.pdf

[3] “A General Language Assistant as a Laboratory for Alignment”, Askell et al. https://arxiv.org/pdf/2112.00861.pdf

[4] “Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression”, Raventós et al. https://arxiv.org/abs/2306.15063

[5] “Learning by distilling context”, Charlie Snell et. al. Charlie Snell, https://arxiv.org/pdf/2209.15189.pdf

[6] https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

[7] https://github.com/uds-lsv/llmft
