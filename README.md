# Telauges Alpha

__Currently, Max-pooling has some issues with cuDNN support, please use basic useage only__

I finally decided to rewrite my coding style for neural networks. This code will be eventually merged to Telauges.

This code is heavily referred to three deep learning library

+ [Pylearn2](https://github.com/lisa-lab/pylearn2)
+ [Blocks](https://github.com/mila-udem/blocks)
+ [theano_lstm](https://github.com/JonathanRaiman/theano_lstm)

## Requirements

General requirements can be achieved by `Anaconda`. More specifically, you need:

+ `python 2.7`
+ `numpy`
+ `scipy`
+ `theano`

## Supported Features

+ Feedfward Layer
   + Identity Layer
   + Tanh Layer
   + Sigmoid Layer
   + ReLU Layer
   + Softmax Layer (classification layer)
   + SVM [TODO]

+ ConvNet Layer
   + Identity Conv Layer
   + Tanh Conv Layer
   + Sigmoid Conv Layer
   + ReLU Conv Layer
   + Max-pooling Layer
   + Same size Max-pooling Layer
   + Flatten Layer

+ Recurrent Neural Network
   + Simple Recurrent Net [TODO]
   + Long-Short Term Memory [TODO]
   + Gated Recurrent Net [TODO]

+ Training Algorithm
   + Stochastic gradient descent (SGD) (Momentum and Nestrov Momentum)
   + Adagrad
   + Adadelta
   + RMSprop
   + Dropout [TODO]

+ Model
   + Feedforward Model
   + Auto-encoder Model
   + Convolutional Auto-encoder Model [TODO]
   + Mixed Layer Model [TODO]
   
## Notes

### NeuralTalk

I'm trying to reproduce the result of [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://cs.stanford.edu/people/karpathy/deepimagesent/).

Here is some of my notes while playing with the dataset. There are descriptions of three datasets are available:
+ Flickr8k
+ Flickr30k
+ Microsoft COCO

The description file `dataset.json` is organized in this way (just a raw description). For each image
```
 -- image info
    |
    -- filename (string)
    -- imgid (int)
    -- sentences (dict)
       |
       -- tokens (dict, in order)
       -- raw (raw sentence)
       -- imgid (int, image that associates with the description)
       -- sentid (int, label of sentence)
       ...
    -- split (string, "train", "val", or "test")
    -- sentids (dict)
```

## Contacts

Yuhuang Hu  
__No. 42, North, Flatland__  
Email: duguyue100@gmail.com
