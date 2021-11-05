from setuptools import setup, find_packages

setup(
  name = 'docformer',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'DocFormer: End-to-End Transformer for Document Understanding',
  author = 'Akarsh Upadhay, Shabie Iqbal',
  author_email = 'akarshupadhyayabc@gmail.com, shabieiqbal@gmail.com',
  url = 'https://github.com/shabie/docformer',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'document understanding',
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'torchvision',
    'transformers',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
