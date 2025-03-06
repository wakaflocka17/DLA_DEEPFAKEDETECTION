# 1. METADATA ANALYSIS
## 1.1 Pre-trained network architectures: MobileNet e Xception
This section provides an overview of two custom neural networks, MobileNet and Xception, with specific attention regarding their architectures and underlying principles. Both architectures uses the concept of depthwise separable convolutions to obtain an optimal balance between computational efficiency and accuracy, although they were designed for different purposes and applications.

### 1.1.1 Mobilenet
MobileNet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications Andrew G. Howard et Al.] is a suite of highly efficient convolutional neural network architectures specifically designed for mobile and embedded vision applications. Its key innovation is the use of depthwise separable convolutions—a type of factorized convolution that splits a standard convolution into two separate operations. In the first stage, the depthwise convolution applies a single filter to each input channel, effectively isolating channel-specific features. In the second stage, the pointwise convolution (a 1×1 convolution) combines these individual outputs to form a new set of features. This two-step process replaces the traditional one-step convolution that both filters and combines inputs, leading to a significant reduction in computational cost and model size without a substantial loss in accuracy.
The following image shows the difference between 
- Standard convolutional layer with batch norm and ReLU on the left.
- Depth wise Separable convolutions with Depthwise
 and Pointwise layers followed by batch norm and ReLU.

<div align="center">
  <img src="images/image-1.png" alt="Mobilenet architecture">
</div>

MobileNet distinguishes itself with its remarkable flexibility. It introduces two global hyperparameters—the width multiplier and the resolution multiplier—which allow developers to fine-tune the trade-off between accuracy, latency, and model size. This adaptability enables the network to be scaled down for resource-constrained environments while still maintaining competitive performance on tasks like image classification, object detection, and beyond.

### 1.1.2 Xception
Xception [Xception: Deep Learning with Depthwise Separable Convolutions Franc¸ois Chollet] is a convolutional neural network architecture that takes the idea of depthwise separable convolutions to its extreme. Instead of relying on complex inception modules, Xception completely decouples the learning of spatial and cross-channel correlations. This is done in two simple steps: first, a depthwise convolution independently extracts spatial features from each channel; then, a pointwise (1×1) convolution fuses these features across channels. This “extreme” formulation (hence the name Xception), short for “Extreme Inception”—leads to a more efficient use of parameters.
A convolution layer attempts to learn filters in a 3D space, with 2 spatial dimensions (width and height) and a channel dimension; thus a single convolution kernel is tasked with simultaneously mapping cross-channel correlations and spatial correlations. This idea behind the Inception module is to make this process easier and more efficient by explicitly factoring it into a series of operations that would independently look at cross-channel correlations and at spatial correlations. The typical Inception module first looks at cross channel correlations via a set of 1x1 convolutions, mapping  the input data into 3 or 4 separate spaces that are smaller than the original input space, and then maps all correlations in these smaller 3D spaces, via regular 3x3 or 5x5 convolutions.

<div align="center">
  <img src="images/image.png" alt="canonical inception module">

  A canonical Inception module (Inception V3).
</div>

<div align="center">
  <img src="images/Screenshot 2025-03-06 162939.png" alt="extreme version of inception module" >

  An “extreme” version of our Inception module, with one spatial convolution per output channel of the 1x1 convolution.
</div>

## 1.2 Differences between the two models
1. Design Motivation and Target Use-Case:
   - MobileNets were primarily designed for mobile and embedded vision applications. They emphasize efficiency by trading off accuracy for lower latency and smaller model size, making them ideal for resource‐constrained environments.
   - Xception reinterprets the Inception module by taking the idea to its extreme. Its motivation is to fully decouple the mapping of spatial correlations from cross-channel correlations using depthwise separable convolutions, aiming for improved performance on large-scale image classification tasks. 
2. Architectural Composition:
   - MobileNets build their architecture almost entirely from depthwise separable convolutions and introduce two hyperparameters—the width multiplier and the resolution multiplier—to flexibly adjust the network’s size and computational cost. 
   - Xception replaces Inception modules with a linear stack of depthwise separable convolutions organized into deeper modules (36 convolutional layers grouped into 14 modules) and incorporates residual connections throughout. 
3. Order of Operations and Activation Placement
   - In MobileNets, each depthwise separable convolution is executed by first applying a depthwise (spatial) convolution and then a pointwise (1×1) convolution—with batch normalization and ReLU applied after each convolution—to capture both spatial and cross-channel correlations. 
   - Xception also uses depthwise separable convolutions but investigates a slightly different strategy: experimental results suggest that omitting the non-linearity (such as ReLU) between the depthwise and pointwise operations may actually yield faster convergence and better performance. This subtle change is part of its “extreme” interpretation of the Inception hypothesis. 
4. Use of Residual Connections
   - MobileNets do not incorporate residual (skip) connections in their basic architecture.
   - Xception makes extensive use of residual connections around its modules to help with training convergence and overall performance. 

These differences reflect the distinct goals of the two models: MobileNets are optimized for efficiency under tight computational constraints, while Xception leverages a more radical factorization of convolutions (with residual connections) to push performance on large-scale tasks.

An in depth view of how the Xception model can be used is seen in the paper done by R. Helaly et al. [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9329302&isnumber=9329288]
