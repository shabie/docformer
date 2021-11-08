# About the files:


```python
1. dataset.py
```
* This file contains the various functions, which are required for the pre-processing of the image as for an example, running the OCR for getting the bounding box of each word, getting the relative distance between the coordinates of the different word, performing the tokenization, and some optional arguments which are required for saving the different features on the disk, and many more!!!


* The code can be modified as per the requirement, as for an example, if we have to perform a specific task of segmenting the image or classifying a document, with some modifications in the ```create_features``` function, it can be achieved.

```python
2. dataset_pytorch.py
```
* This file inherits the functions from the ```dataset.py```, however it creates a Dataset object of the file stored in the disk, and the function can be modified for some augmentations as well
* The Dataset object is required for the purpose of training the model in PyTorch, however in TensorFlow, the numpy version would work instead of Dataset and DataLoader object

```python
3. modeling.py
```
* This file is the brain of everything in this repo, the file contains the various functions, which have been written with least approximation in mind, and as close to the paper, it contains the ```multi-head attention```, the various embedding functions, and a lot of stuffs, which are mentioned in the paper. In order, to understand this file properly, one of the suggestion is to, open the code and the paper side by side, and that would work.
* And, for the task specific requirements, one can import ```DocFormerEncoder```, and attach one head for the task-specific requirement, however, the last function ```Decoder``` is a work in progress, which is for the Image Reconstruction purpose as mentioned in the paper (in the pre-training part)
