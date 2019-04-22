# CNN model compression using Transfer Learning
&copy; Ashis Ravindran *&*  Enrique Fita Sanmartin, Universität Heidelberg.

### Abstract

Training a deep neural network demands a huge amount of computational resources, and in the real-world scenario, the trained cumbersome model would drastically hinder the performance in the deployment stage in terms of computation and speed, despite its proven decent accuracy. Thus it is important to efficiently distil the knowledge from the large model (teacher) to get a smaller one (student) not trading too much on the original accuracy. 
This project ponders over the possibilities of **inter-network training**. We compare how different configurations improve the transfer of knowledge (a.k.a Dark Knowledge) from ensemble/teacher model to smaller/student model. We propose a new configuration for inter-model training called the **Pyramid**. All experiments were carried out using VGG19 & VGG19 based sub-architectures, trained using STL10 & CIFAR100. The newly derived model using the proposed approach has 10x fewer parameters and 4x faster than VGG19.
Read the full report [here](docs/report_Pyramid_Intership.pdf)!


### Pyramid Scheme

We propose a set of training schemes, which rely on transfer learning techniques, in compressing a neural network. We call this training strategy: **The Pyramid**. The two variations are explored in the code base, viz.  *Forward scheme* and *Backward scheme*.
Various VGG19 sub models SVGG17, SVGG14, SVGG11, SVGG8 and SVGG5 were created to experiment for different compressing levels. 
The compressing result shows that the newly proposed pyramid configuration achieves the state of art classification accuracy for a compressed smaller model trained from scratch!
The experiments carried out to explore the compression also compares different loss functions and their effect on the *inter-model* training.
Again, Read the full report [here](docs/report_Pyramid_Intership.pdf)!

### Caveat

As you might see, the code base is a bit clumsy. *But you get the idea...!*
Code was developed in `Python 3.5` with `PyTorch` version 0.2. Yep, its from 2017...

### Results

Using VGG based sub models for compression, SVGG5 (model with 5 conv layers and one FC layer) gave a whooping **91%** accuracy *(compared to 72.8% when trained from scratch)* on STL10 and **74.71%** accuracy (compared to 61.56% when trained from scratch)* on CIFAR 100.
Again, Read the full report [here](docs/report_Pyramid_Intership.pdf)!

### Who are we

We are Enrique Fita Sanmartin & Ashis Ravindran, Master students of Scientific Computing at University of Heidelberg, Germany. 
Contact us: `enfisan@alumni.uv.es`,`ashis.r91@gmail.com` 
