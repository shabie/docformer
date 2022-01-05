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
* And, for the task specific requirements, one can import ```DocFormerEncoder```, and attach one head for the task-specific requirement, however, the last function ```ShallowDecoder``` is a work in progress, which is for the Image Reconstruction purpose as mentioned in the paper (in the pre-training part)

```python
4. modeling_pl.py
```
* This file is basically, for the parallelization of the training and validation part, so that the utilization of multiple GPUs becomes easy
* This file contains the integration of PyTorch Lightening, with the DocFormer model, so that the coding part becomes less and the task specific things can be done.
* For task specific requirements, one can modify the ```Model``` class, with the modification being in the ```training``` and ```validation``` step, where the specific loss functions can be integrated and thats it!!!


```python
5. train_accelerator.py
and 
6. train_accelerator_mlm_ir.py
```
* These files are also codes for the purpose of training so that the coding requirements becomes less.
* The only thing is that, these code inherits the ```Accelerator``` of "Hugging Face", for the purpose of Parallelization of the task to multiple GPUs
* ```train_accelerator.py``` contains the function of running the code of Pre-training the model with ```MLM``` task
* ```train_accelerator_mlm_ir.py``` contains the function of running the code of Pre-training the model with ```MLM and Image Reconstruction (IR)``` task, however we are thinking of making a file which contains the options of training according to specific task



```python
7. utils.py
```

* File, which contains the utility function for performing the unsupervised task of Text Describe Image (as mentioned in the paper: DocFormer)

How to use it?
* Let us assume, that all the entries of the dataset have been stored somewhere.
* Now, we can get the length of the entries, and that has to be passed to the function ```labels_for_tdi```, which would give the arguments as well as the labels, now, iterate through each of the arguments, and for ith entry, create a new entry in the dicitionary (data format for docformer, refer to dataset.py, create_features function), and map it to the resized_scaled_img of arr[i], 

* i.e in terms of pseduocode,
Assume, that  
```python

l-> list of dictionary format, data points of docformer
d_arr, labels = labels_for_tdi(n)
for i, j in enumerate(d_arr):
    l[i]['d_resized_scaled_img'] = l[j]['resized_scaled_img']
    l[i]['label_for_tdi'] = labels[i]
```

And then, the rest follows by passing the argument `use_tdi`, for the model

Using it with the model:
For the purpose of integrating TDI with model, the following instruction would be helpful:

Let us assume, we want to do MLM + IR + TDI:

1. As for the first and the third task, attach a head, and forward propagate the data, calculate the weighted loss of these two task, and store it
2. In case of the second task, you have to forward propagate it again with the same dataset, but with the argument use_tdi = True, and calculate the binary cross entropy loss with the `label_for_tdi` key, and add the weighted sum of it to the stored loss, and then backpropagate it
