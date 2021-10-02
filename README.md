# DocFormer - PyTorch

![docformer architecture](images/docformer-architecture.png)

Implementation of [DocFormer: End-to-End Transformer for Document Understanding](https://arxiv.org/abs/2106.11539), a multi-modal transformer based architecture for the task of Visual Document Understanding (VDU) 📄📄📄.

DocFormer is a multi-modal transformer based architecture for the task of Visual Document Understanding (VDU). In addition, DocFormer is pre-trained in an unsupervised fashion using carefully designed tasks which encourage multi-modal interaction. DocFormer uses text, vision and spatial features and combines them using a novel multi-modal self-attention layer. DocFormer also shares learned spatial embeddings across modalities which makes it easy for the model to correlate text to visual tokens and vice versa. DocFormer is evaluated on 4 different datasets each with strong baselines. DocFormer achieves state-of-the-art results on all of them, sometimes beating models 4x its size (in no. of parameters).

The official implementation was not released by the authors.

## Install


## Usage

See `examples` for usage.

##  License

MIT

## Maintainers

- [uakarsh](https://github.com/uakarsh)
- [shabie](https://github.com/shabie)

## Contribute


## Citations

```bibtex
@misc{appalaraju2021docformer,
    title   = {DocFormer: End-to-End Transformer for Document Understanding},
    author  = {Srikar Appalaraju and Bhavan Jasani and Bhargava Urala Kota and Yusheng Xie and R. Manmatha},
    year    = {2021},
    eprint  = {2106.11539},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
