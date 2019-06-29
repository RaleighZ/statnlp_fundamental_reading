# CNN: From LeNet to DenseNet
by zhijiang

## Motivation
Key principles for building NNs for CV:

- **Translation Invariance:** The system should respond similarly to the same object regardless of where it appears in the image.
- **Locality:**: The system should focus on local regions, without regard for what else is happening in the image at greater distances. 
- **All learning depends on imposing inductive bias:** here we deduce the convolution by incorporating these two assumptions.
![](./figs/MLP.png){width="400px" height="50px" align="center"}
For any given location (i, j) in the hidden layer h[i, j] of a MLP, we compute its value by summing over pixels in x, centered around (i, j) and weighted by V[i, j, a, b].
	- **Translation Invariance:**  This is only possible if V doesn't actually depend on (i, j):![](./figs/invariance.png){width="200px" height="50px" align="center"}. This reduces the number of parameters by a factor of 1 million (for 1 megapixel image) since it no longer depends on the location within the image.
	- **Locality:** we should not look very far away from (i, j) in order to glean relevant information to assess what is going on at h[i, j]: ![](./figs/locality.png){width="200px" height="50px" align="center"}. Outside some range |a|, |b| > delta, V[a, b] = 0.



## Concepts

- **Convolutions for Images:** 
	- **Kernel (Filter)**:  ![](./figs/Conv.png){width="250px",  height="100px", align="center"}.
	- **Edge Detection**: The middle 4 columns are black and the rest are white. ![](./figs/edge.png){width="150px",  height="100px", align="center"}.
We construct a kernel with a height of 1 and width of 2 ([1, -1]). As the result here, we will detect **1** for the **edge from white to black** and **-1** for the **edge from black to whilte**. The result of the outputs are 0.
![](./figs/result.png){width="150px",  height="100px", align="center"}.
In practice, we learn the kernel by looking at the (input, output) pairs only. We initialize kernel as a random arry. Next, in each iteration, we will use the squared error to compare Y and the output, then calculate the gradient to update the weight.



- **Padding and Stride**
	- **Output shape:** assume the input shape in Nh x Nw and the kernel window is Kh x Kw, then the output shape is:![](./figs/shape1.png){width="300px",  height="50px", align="center"}.
	- **Motivation:**
		- Since kernels generally have width and height greater than 1. After applying many successive convolutions, we will wind up with an output that is much smaller than our input. *Padding* handles this issue.
		- In some cases, we want to reduce the resolution drastically, if say we find our originall input resolution to be unwieldy.
	- **Padding:** assume padding size is Ph and Pw. In many case, we will want to set Ph = Kh - 1 and Pw = Pw - 1 to give the input and output the same height and width.  ![](./figs/shape2.png){width="300px",  height="40px", align="center"}. CNN commonly use kernels with **odd** height and width values, so we can preserve the spatial dimensionality while padding with the same number of rows (columns) on top and bottom.
	![](./figs/pad.png){width="250px",  height="100px", align="center"}.
	- **Stride:** for computational efficiency or downsample, we have stride: ![](./figs/shape3.png){width="300px",  height="40px", align="center"}. 
	![](./figs/stride.png){width="250px",  height="100px", align="center"}.
	
	
- **Multiple Channels:** we sum over the channels.
![](./figs/channel.png){width="500px",  height="200px", align="center"}.
	- **Multiple Output Channels:**it turns out to be essential to have multiple channels at each layer. Actually we increase the channel dimension as we go higher up in the NN, typically downsompling to trade off spatial resolution for greater channel depth. You could think each channel as responding to some different set of features (feature map)
	- **Shape of the kernels:** assume Ci and Co are the number of input and output channels. For each output channel we create Ci x Kh x Kw kernels. We concat them on the output channel dimension, so that the shape of the kernel is Co x Ci x Kh x Kw. 
	
- **Pooling:** dual purpose"
	- **Mitigate the sensitivity of convolutional layers to location:** in reality, objects hardly ever occur exactly at the same place. Assume in edge detection, the conv layer input is X and the pooling output is Y. Whether or not the values of X[i, j] and X[i, j+1] are different, or X[i, j+1] and X[i, j+2] are different, the max pooling (2 x 2) layer outputs all include Y[i, j]=1. We can still detect if the pattern recognized by the conv layer moves no more than 1 element in height and width.
	- **Spatially Downsample:** as we process images, we want to gradually reduce the spatial resolution of our hidden representations, aggregating information so that the higher up we go in the network, the larger the **receptive field**  to which each hidden node is sensitive. By gradually aggregating information, yielding coarser and coarser maps, we accomplish this goal of ultimately learning a global representation, while keeping all of the advantages of convolutional layers at the intermediate layers of processing.

- **1 x 1 Convolutional Layer:** 1 X 1 convolution loses the ability to recognize patterns consisting of interactions among adjacent elements in the height and width dimensions. **The only computation of the 1 x 1 convolution occurs on the channel dimension**
![](./figs/1by1.png){width="500px",  height="200px", align="center"}.
Each element in the output is derived from a linear combination of elements *at the same position* in the input image. You could think the 1 x 1 as constituing a fully-connected layer applied at every single pixel location to transform the Ci input values into Co output values. This layer requires Co x Ci weights.


## CNN Architectures

- **LeNet**

- **AlexNet**

- **VGG**

- **NiN**

- **GoogleNet**

- **ResNet**

- **DenseNet**





## Useful Links
1. 理解 Word2Vec 之 Skip-Gram 模型: https://zhuanlan.zhihu.com/p/27234078
2. Word2Vec数学推导： https://zhuanlan.zhihu.com/p/53425736
3. What is GloVe: https://towardsdatascience.com/emnlp-what-is-glove-part-i-3b6ce6a7f970
4. The Annotated Transformer: http://nlp.seas.harvard.edu/2018/04/03/attention.html

