by guoshun

## Motivation

**CNN is unable to  handle non-Euclidean data by shared tunable kernels, While GNN can operate on graph domain. Let's see the details of why we use GNN**

Firstly, let's see the convolutionals on Euclidean data (grids, sequences). The update for sequences by RNN and grids by CNN can be formulated as. 

![gnn-motivation-f](figs/gnn-motivation-f.jpg)



Here is a demostration of 3X3 convolution on grid data and the udpate [1]:

## ![cnn-motivation](figs/cnn-motivation.png)

What if our data looks like this [1] ? such as data from social networks, word-wild-web, knowledge graphs, telecommunication network. Can we still use RNN or CNN as a convolution operator?

![gnn-motivation2](figs/gnn-motivation2.png)



Obviously we can not use RNN or CNN directly to model graph data. Let's give the definitions of graphs. G=(V, E) indicates a set a nodes V: {v_i} and edges E: {(v_i, v_j)}, where a edge (v_i, v_j) represents a connection between node v_i and node v_j. We also define define adjacent matrix A , A_{ij} = 1 if (v_i, v_j) is in the set E.

### Question:

1. Can we feed the GCN network into deep FFN and what are the problems ?   

2. What are the relations between GCN, CNN and RNN ? Can GCN be more superior than tree-LSTM? 

   

Let's X_{in} = [X, A] denote the input of deep FNN, X and A indicate feature matrix and ajacent matrix, respectively.  The problems are[1]:

![gnn-mlp](figs/gnn-mlp.png)

- Huge number of parameters O(N), N indicates the length of feature matrix.

- Needs to be re-trained if number of nodes changes.

- Does not generalize across graphs

  

Let's X_{in} = [X, A] denote the input of deep FNN, X and A indicate feature matrix and ajacent matrix, respectively.  The problems are:

![gnn-def-1](figs/gnn-def-1.jpg)

![gnn-def-2](figs/gnn-def-2.jpg)

Or treat self-connection in the same way.

![gnn-def-3](figs/gnn-def-3.jpg)

![gnn-def-4](figs/gnn-def-4.jpg)



![gnn-def-5](figs/gnn-def-5.png)

## Graph Types:

***Directed graphs***: There is a relation between two nodes. For example,  in a knowledge graph where teh edge starts form the head entity and ends at the tail entity.  

Undirected edge can be treated as two directed edges, like graph constructed by dependency tree in relation extraction in "Zhang et al. Graph Convolution over Pruned Dependency Trees Improves Relation Extraction, EMNLP 2018". 

***Heterogeneous graphs***:  There are several kinds of nodes and each type of nodes is converted into a one-hot feature, for example, "Tu et al. Multi-hop Reading Comprehension across Multiple Documents by
Reasoning over Heterogeneous Graphs, ACL 2019"

***Graphs with edge information***: Two ways to handle this graph: further splitting and relational-GCN, which uses different type of edges [3].

![relational-GCN](figs/relational-GCN.png)

## Propogation Steps

This Figure is copied from [2].

![gnn-progation](figs/gnn-progation.png)

1. Convolutional aggreator: 

2. Attention aggregator: 

3. Spectral methods: 

4. Spatial methods:

   

## Variants of GNN

The variants of GNN include GCN,  GAT, Gated GNN and Graph LSTM, and the main difference lies in their aggregator and updater.

- Aggregator:  Gathering information from other nodes

- Updater:  Updating node's hidden states

  

For details please refer to Table2  of paper [2] Graph Nerual Networks: A review of methods and Applications.

![gcn-category](figs/gcn-category.png)

## Further Questions:

1. Can GCN be trained very deep without over-smoothing caused by many convolution layers ?

   1. Thoretically, GCN model is actually is a special form of Laplacian smoothing, and repeatly applying the smoothing will cause  each node **converge to the same values** :  Deeper Insights into Graph Convolutional Networksfor Semi-Supervised Learning, AAAI 2019.
   2. Gating machanism is able to alliviate Laplacian smoothing of GCN. "Tu et al. Multi-hop Reading Comprehension across Multiple Documents by Reasoning over Heterogeneous Graphs, ACL 2019"
   3. From the gradient viewpoint, GCN can be deeper with highway connections between layers referring to **ResNet** and **Densenet**

   ![over-smoothing](../../reading%20group/gnn/figs/over-smoothing.png)

   

2. How GCN handles the long-range depency problem?

   1. Dependency-Tree

   2. Graph-LSTM

      

## SOTA

[1] Simplifying Graph Convolutional Networks (SGC), ICML 2019

***Motivatin***: can we reduce the excess complexity and redundant computation of GCN?

**Insights: A linear model (SGC) is sufficient on many graph tasks.**

![SGC](figs/SGC.png)

SGC performan on par with or better than GCN across 5 tasks including 14 datasets for graph classification, text classification, semi-supervised user geolocation, relation extraction and zero-shot image classification.

[2] Position-aware Graph Neural Networks, ICML 2019

Motivation: Learning node embedding that capture a node's position within the broader graph strucgure is crutial for many prediction task. However, existing work have limited power in capturing the position/location of a given node with respect to all other nodes of the graph.

![position](figs/position.png)

[3] LatentGNN: Learning Efficient Non-local Relations for Visual Recognition, ICML 2019

Motivation:  to model non-local  context relations for capturing  **long-range dependencies** in feature representations.

![latentGCN](figs/latentGCN.png)

[4] Mixhop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing

![mixhop](figs/mixhop.png)

## Useful Link

[1] Graph Convolution Networks :  <https://tkipf.github.io/graph-convolutional-networks/>  by Thomas Kipf, ICLR 2017.

[2] Graph Nerual Networks: A review of methods and Applications.<https://arxiv.org/pdf/1812.08434.pdf> by Jie Zhou, and Maosong Sun.

[3] Modeling Relational Data with Graph Convolutional Networks, <https://arxiv.org/pdf/1703.06103.pdf>, by Michael Schlichtkrull, and MaxWelling, *European Semantic Web Conference*. Springer, Cham, 2018

