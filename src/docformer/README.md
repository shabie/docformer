# About the files:


```python
1. dataset.py
```
* This file contains the various functions, which are required for the pre-processing of the image as for an example, running the OCR for getting the bounding box of each word, getting the relative distance between the coordinates of the different word, performing the tokenization, and some optional arguments which are required for saving the different features on the disk, and many more!!!


* The code can be modified as per the requirement, as for an example, if we have to perform a specific task of segmenting the image or classifying a document, with some modifications in the "create_features" function, it can be achieved.

```python
2. dataset_pytorch.py
```
* This file inherits the functions from the "dataset.py", however it creates a Dataset object of the file stored in the disk, and the function can be modified for some augmentations as well
* The Dataset object is required for the purpose of training the model in PyTorch, however in TensorFlow, the numpy version would work instead of Dataset and DataLoader object

```python
3. modeling.py
```

