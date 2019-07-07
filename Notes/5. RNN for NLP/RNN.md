# Recurrent Neural Network (RNN) for NLP
## Course Information
- Video: https://www.youtube.com/watch?v=6lmHY8qckm0
- Slide: http://www.phontron.com/class/nn4nlp2019/assets/slides/nn4nlp-05-rnn.pdf
- Materials: Chapter 14-15 in Neural Network Methods for Natural Language Processing by Yoav Goldberg
## Motivation
- Q1: Can we design a tool to process input of variable length, i.e., the sequence
- Q2: Can we construct a feature extractor that can attend to the order of input sequence and relation of different parts of sequence which locates far away from each other 


## Concept
- Long Range Dependency (LRD)
  - An **element** in a sequence is related to another element that is far away from **itself**.
  - An example: 'An **element** in a sequence is related to another element that is far away from **itself**'
- Memory mechanism

## Basic Method
### LRD Modeling Issues
- In what sense, can we say a model have considered the effect of Long Range Dependency and try to fix it?
  - The model tries to model the conditional Probability between 'dependentER' and 'dependentEE'
- N-gram to Model LRD
  - Assume the Markov Property, i.e., restricting the dependency range to a fix length
  - Fancy Way: Singe layer CNN: n-gram filter.
- Longer distance? Unlimited Distance?
  -  increasing N 
     - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/CNN_LRD.png)
  -  N-gram of N-gram, i.e., stacked CNN  
     - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/STACK_CNN_LRD.png)
  - Memory mechanism
    - encode passing input as memory
    - Construct RNN from CNN
      - RNN as memory CNN with kernel size = 2
      - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/RNN_LRD.png)
    
  
## What is RNN:
- RNN is an NN that generates output vector based on previous hidden state and current input. The parameters of RNN are **shared** along the input 
  - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/rnn_as%20ffn.png)
  - ![](https://latex.codecogs.com/gif.latex?\mathrm{RNN}^{\star}\left(\boldsymbol{x}_{\mathbf{1}&space;:&space;\boldsymbol{n}}&space;;&space;\boldsymbol{s}_{\boldsymbol{0}}\right)=\boldsymbol{y}_{\mathbf{1}&space;:&space;\boldsymbol{n}})
  - ![](https://latex.codecogs.com/gif.latex?y_{i}=O\left(s_{i}\right))
  - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{s}_{\boldsymbol{i}}=R\left(\boldsymbol{s}_{\boldsymbol{i}-\mathbf{1}},&space;\boldsymbol{x}_{\boldsymbol{i}}\right))
 

## How to train RNN
- BackPropagation Through Time (BPTT)
  - unroll RNN and view RNN as a special deep feedforward NN.
- Gradient Vanishing Issues  (backforward)
  - dual problem (forward): the input has a little impact on the ouput if they are far from each other, i.e. long dependency issue. Output **tends to forget previous state**.
  - What is Gradient Vanishing
    - gradient of loss at W:  G(w) = G_1(W) + G_2(W) + ... + G_n(W), where 1,2, n are time index
    - gradient of loss at RNN parameter at time index i
    - has little effect on loss
  - Why
    - RNN as a deep NN
    - gradient as consecutive Matrix muplication, exploring or vanishing
  - How
    - Improve memory mechanism
  
## Variants of RNN
- Gated Memory
  - Highway Connection (element-wise gating)
    - ![](https://latex.codecogs.com/gif.latex?s^{\prime}&space;\leftarrow&space;g&space;\odot&space;x&plus;(1-g)&space;\odot(s))
    - Have direct connection from input to output
 - Long Short-Term Memory (LSTM)
    - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/LSTM.png)
    - ![](https://latex.codecogs.com/gif.latex?s_{j}=R_{\text&space;{&space;LSTM&space;}}\left(s_{j-1},&space;x_{j}\right)=\left[c_{j}&space;;&space;h_{j}\right])
    - ![](https://latex.codecogs.com/gif.latex?c_{j}=f&space;\odot&space;c_{j-1}&plus;i&space;\odot&space;z)
    - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{h}_{\boldsymbol{j}}=\boldsymbol{o}&space;\odot&space;\tanh&space;\left(\boldsymbol{c}_{\boldsymbol{j}}\right))
    - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{i}=\sigma\left(\boldsymbol{x}_{j}&space;\boldsymbol{W}^{\boldsymbol{x}&space;i}&plus;\boldsymbol{h}_{\boldsymbol{j}-\mathbf{1}}&space;\boldsymbol{W}^{\boldsymbol{h}&space;i}\right))
    - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{f}=\sigma\left(\boldsymbol{x}_{\boldsymbol{j}}&space;\boldsymbol{W}^{\boldsymbol{x}&space;f}&plus;\boldsymbol{h}_{\boldsymbol{j}-\mathbf{1}}&space;\boldsymbol{W}^{\boldsymbol{h}&space;\boldsymbol{f}}\right))
    - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{o}=\sigma\left(\boldsymbol{x}_{\boldsymbol{j}}&space;\boldsymbol{W}^{\boldsymbol{x}&space;\boldsymbol{o}}&plus;\boldsymbol{h}_{\boldsymbol{j}-\mathbf{1}}&space;\boldsymbol{W}^{\boldsymbol{h}&space;\boldsymbol{o}}\right))
    - ![](https://latex.codecogs.com/gif.latex?z=\tanh&space;\left(x_{j}&space;W^{x&space;z}&plus;h_{j-1}&space;W^{h&space;z}\right))
    - ![](https://latex.codecogs.com/gif.latex?y_{j}=O_{\mathrm{LSTM}}\left(s_{j}\right)=h_{j})
    - cj is the memory component, hj is the hidden state component. 
    - Three gates, **i** , **f** , and **o**, controlling for **input**, **forget**( which does not exist in original version), and **output**.
- Gated Recurrent Unit (GRU), simple gating mechanism
    - ![](https://latex.codecogs.com/gif.latex?s_{j}=R_{\mathrm{GRU}}\left(s_{j-1},&space;x_{j}\right)=(1-z)&space;\odot&space;s_{j-1}&plus;z&space;\odot&space;\tilde{s_{j}})
    - ![](https://latex.codecogs.com/gif.latex?z=\sigma\left(x_{j}&space;W^{x&space;z}&plus;s_{j-1}&space;W^{s&space;z}\right))
    - ![](https://latex.codecogs.com/gif.latex?\boldsymbol{r}=\sigma\left(\boldsymbol{x}_{\boldsymbol{j}}&space;\boldsymbol{W}^{\boldsymbol{x}&space;\boldsymbol{r}}&plus;\boldsymbol{s}_{\boldsymbol{j}-\mathbf{1}}&space;\boldsymbol{W}^{\boldsymbol{s}&space;\boldsymbol{r}}\right))
    - ![](https://latex.codecogs.com/gif.latex?\tilde{s_{j}}=\tanh&space;\left(\boldsymbol{x}_{j}&space;\boldsymbol{W}^{\boldsymbol{x}&space;s}&plus;\left(\boldsymbol{r}&space;\odot&space;\boldsymbol{s}_{j-\mathbf{1}}\right)&space;\boldsymbol{W}^{s&space;g}\right))
    - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/gru.png)
- Explanation from Gradient Perspective

## Bidrectional Issue
- Although RNN relaxs the markov assumption, it retricts the parts of sequence that influence current output to either the parts that follows or the part shows up before. However, the following words may also be useful for prediction/better feature extraction.
- Encode Information from two direction
  - ![](https://github.com/RaleighZ/statnlp_fundamental_reading/blob/master/Notes/5.%20RNN%20for%20NLP/fig/bi-rnn.png)
- Misc.
  - Deep-Bi-RNN
    1. concatenation of two deep RNNs. -> ELMO
    2. the output sequence of one biRNN is fed as input to another -> stacked RNN
    - Empirically speacking, seems the second is better thatn the first one, 
## Application Issue
- discussion
## Problems/Discussion
- What is **long range dependency**, how to define, how to extract, how to measure in NLP
- Why we need a forget gate in LSTM
- RNN as A special case of memory-CNN, ANY extension?
- Link between RNN, CNN and Transformer

## SOTA
    
