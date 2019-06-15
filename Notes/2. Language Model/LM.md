# Language Modeling
## Course Information
- Video: https://www.youtube.com/watch?v=0pycTk-JFk8
- Slide: http://www.phontron.com/class/nn4nlp2019/assets/slides/nn4nlp-02-lm.pdf
- Materials: [Chapter 9 in Neural Network Methods for Natural Language Processing by Yoav Goldberg](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Language%20Model/Goldberg_ch9_Language%20Model.pdf)
## Motivation
- Q1: How to decide which sentence is natural, which is not? Or how to judge a sentence is good or not?
Ans: possibility, to evaluate a sentence by calculating its possibility
- Q2: How to calculate the possibility of a given sentence?
- Q3: What can we do with language model?


## Concept
- [Chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)): Consider an indexed collection of random variables ![](https://latex.codecogs.com/gif.latex?X_{1},\ldots&space;,X_{n}). To find the value of this member of the joint distribution, we can apply the definition of conditional probability to obtain:

![](https://latex.codecogs.com/gif.latex?\mathrm&space;{P}&space;(X_{n},\ldots&space;,X_{1})=\mathrm&space;{P}&space;(X_{n}|X_{n-1},\ldots&space;,X_{1})\cdot&space;\mathrm&space;{P}&space;(X_{n-1},\ldots&space;,X_{1}))
- [Markov Property](https://en.wikipedia.org/wiki/Markov_property)
- [Entropy Rate]()
    - the **entropy rate** of a [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process "Stochastic process") is, informally, the time density of the average information in a stochastic process.
    - ![](https://latex.codecogs.com/gif.latex?H(X)&space;=&space;\lim_{n&space;\to&space;\infty}&space;\frac{1}{n}&space;H(X_1,&space;X_2,&space;\dots&space;X_n))
- [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy): 
    - ![](https://latex.codecogs.com/gif.latex?H(p,q)=\operatorname&space;{E}&space;_{p}[-\log&space;q])
    - In classification problems, maximizing the likelihood is the same as minimizing the cross entropy.
    - Log likelihood: ![](https://latex.codecogs.com/gif.latex?log\prod&space;_{i}q_{i}^{Np_{i}}&space;=&space;N\sum&space;_{i}p_{i}\log&space;q_{i}&space;=&space;N&space;H(p,&space;q))



## Basic Method
- LM object:
    - How to calculate probability: ![](https://latex.codecogs.com/gif.latex?P\left(w_{1&space;:&space;n}\right)=P\left(w_{1}\right)&space;P\left(w_{2}&space;|&space;w_{1}\right)&space;P\left(w_{3}&space;|&space;w_{1&space;:&space;2}\right)&space;P\left(w_{4}&space;|&space;w_{1&space;:&space;3}\right)&space;\ldots&space;P\left(w_{n}&space;|&space;w_{1&space;:&space;n-1}\right))
- Evaluation: 
    - Calculate perplexity (the lower, the better) on the evaluation corpus
    - **The equvalence of perplexity and cross entropy**
        - Language as a stochastic model, use entropy rate to measure the uncertainty
        - Perplexity is equal to entropy rate
- n-gram LM
    
    - N-order Markov Assumption: ![](https://latex.codecogs.com/gif.latex?P\left(w_{i&plus;1}&space;|&space;w_{1&space;:&space;i}\right)&space;\approx&space;P\left(w_{i&plus;1}&space;|&space;w_{i-n&space;:&space;i}\right))
    - Esimation of the conditional probability
  - Some issues
    - The sparse distribution of features
    - Probability of some features will be 0
    - Smoothing: to solve the zero count issue
  - Weakness
    - Cannot handle long-term dependency because the window of context is limited (n)
    - Word counting method will have feature explosion issues when n increases 
    - Lack of generalization across contexts

- Neural models
    - Try to address 
        - Similar words issues: by word embeddings
        - Feature explosion issue: 
          - Number of features increases polynomially in n-gram model: |V|^n
          - Number of parameters incresese linearly with n
    - A Neural Probabilistic Language Model by Bengio et al. <sup>1</sup>
      - ![](https://latex.codecogs.com/gif.latex?\hat{y}=P\left(w_{i}&space;|&space;w_{1&space;:&space;k}\right)=L&space;M\left(w_{1&space;:&space;k}\right)=\operatorname{softmax}\left(h&space;W^{2}&plus;b^{2}\right))
      - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{h}=g\left(\boldsymbol{x}&space;\boldsymbol{W}^{\mathbf{1}}&plus;\boldsymbol{b}^{\mathbf{1}}\right))
      - ![](https://latex.codecogs.com/gif.latex?x=\left[v\left(w_{1}\right)&space;;&space;v\left(w_{2}\right)&space;;&space;\ldots&space;;&space;v\left(w_{k}\right)\right])
      - ![](https://latex.codecogs.com/gif.latex?v(w)=E_{[w]})
      - ![](https://latex.codecogs.com/gif.latex?w_{i}&space;\in&space;V&space;\quad&space;E&space;\in&space;\mathbb{R}^{|V|&space;\times&space;d_{w}}&space;\quad&space;\boldsymbol{W}^{\mathbf{1}}&space;\in&space;\mathbb{R}^{k&space;\cdot&space;d_{w}&space;\times&space;d_{\mathrm{hid}}})
    - How
      - word embed can encode context information; simliar context yields similar embedding and thus can assign high probability to unseen sentences which have similar context.
      - n -> n+1, ![](https://latex.codecogs.com/gif.latex?\boldsymbol{W}^{\mathbf{1}}): ![](https://latex.codecogs.com/gif.latex?k&space;\cdot&space;d_{\mathrm{w}}&space;\times&space;d_{\mathrm{hid}}) -> ![](https://latex.codecogs.com/gif.latex?(k&plus;1)&space;\cdot&space;d_{\mathrm{w}}&space;\times&space;d_{\mathrm{hid}})
## Related SoTA Work
- [ELMo](https://aclweb.org/anthology/N18-1202) (**E**mbeddings from **L**anguage **Mo**dels)
    - Generate General **Pre-trained** **Contextualized** Word Representation from LM
        - Contextualized
        - Why LM
            - 'Unsupervised' Task, does not require any human annotations
            - Nearly Unlimited data
            - Models trained on the large corpus can generate sentences of an unexpectedly high quality
          
    
    - Model Details
    ![image](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Language%20Model/taglm.png)
        - char-CNN: mitigate OOV
        - stacked bi-LSTM
        - tie parameter in token representation and Softmax layer in the forward and backward directions
    - Deep Bi-directional LM
        - bi-directional LM:
        ![](https://latex.codecogs.com/gif.latex?\begin{array}{l}{\sum_{k=1}^{N}\left(\log&space;p\left(t_{k}&space;|&space;t_{1},&space;\ldots,&space;t_{k-1}&space;;&space;\Theta_{x},&space;\vec{\Theta}_{L&space;S&space;T&space;M},&space;\Theta_{s}\right)\right.}&space;\\&space;{\quad&plus;\log&space;p\left(t_{k}&space;|&space;t_{k&plus;1},&space;\ldots,&space;t_{N}&space;;&space;\Theta_{x},&space;\widetilde{\Theta}_{L&space;S&space;T&space;M},&space;\Theta_{s}\right)&space;)}\end{array})
        - Deep: stacked bi-LSTM
     - Leakage issue in Deep Bi-directional LM?
- [BERT](https://arxiv.org/pdf/1810.04805.pdf) (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers)
    - [Transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)<sup>3</sup>
    - Masked LM
        - Masked some words out in a sentence and predict it
        - Trick
            - Not all words with [MASK] will be merely represented by [MASK]. Instead, 10 percent of words [MASK] will be shown in their original form, like 'dog' for 'dog', 20 percent of them will be replaced by random word, i.e. 'apple' for 'dog' and the others remain [MASK]. 
            - This training tricks mitigate the **unintended bias** resulting from special mark. That is to say, similar tricks can be reused if we have to introduce special mark in the model
    - Next sentence prediction
        - Text-pair classification, which determines whether sentence A is followed by sentence B, semantically or logically
        - When choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus.
    - Model Details
    ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Language%20Model/bert.png)
        - WordPiece embeddings
        - learned positional embeddings
        - Stacked Tranformer encoder
        - Trained with masked LM and next sentence prediction

## Appendix
- Neural Networks Training tricks
    - Shuffling data：reduce bias from the order of training data
    - Optimization methods issues <sup>3</sup>
        - SGD with momentum
        - Adam
    - Early stopping and learning decay
    - Dropout
    - Batching to enhance efficiency
    - Some notes: 
        - "Adam is **usually** fast to converge and stable"
        - "SGD tends to do well in terms of generalization"
        - "Should use learning rate decay"
## Useful Links
1. Bengio et al., A Neural Probabilistic Language Model, 2013: http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
2. Vaswani et al., Attention is All You Need, 2017: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
3. An overview of gradient descent optimization algorithms: http://ruder.io/optimizing-gradient-descent/
