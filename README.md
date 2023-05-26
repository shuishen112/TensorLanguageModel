# TensorTrainLanguageModel

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-orange)](https://pytorch.org/)

TensorTrainLanguageModel is a language model using tensor train implemented in pytorch. We also include the text classification tasks in the folders: ./classficaton_models

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

TensorTrainLanguageModel is designed to apply tensor train to language modeling with pytorch. This project provides TTLM to wikitext2 and ptb dataset. In addition, we also apply TTLM to text classfication tasks. 



## Installation

1. Clone the repository:

git clone https://github.com/<YOUR_GITHUB_USERNAME>/<PROJECT_NAME>.git

2. Change to the project directory:

cd <PROJECT_NAME>

3. Install the required dependencies:

## Usage

To use <PROJECT_NAME>, follow these steps:

<USAGE_INSTRUCTIONS>

### Classification results:


| model | MR | CR | SUBJ | MPQA | SST-2 | 
| :-----| ----: | :----: |:----: | :----: |:----: |
| word2vec | 0.7281 | 0.7338  | 0.8915 | 0.8142 | 0.7666 | 
| CNN |  0.7473| 0.804| 0.9095 |  0.8237| 0.7941|
| RNN | 0.7548 | 0.7881 | 0.914 |  | 

### Example

<EXAMPLE_OF_USING_YOUR_PROJECT>

## Contributing

Contributions are welcome! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name, such as `feature/my-awesome-feature`.
3. Make changes and commit them with a clear and concise commit message.
4. Push your changes to the branch.
5. Create a pull request, describing the changes you've made and why they should be merged.

## License

<PROJECT_NAME> is released under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE` for more information.

## Acknowledgements

- Thank you to <ANY_CONTRIBUTORS_OR_RESOURCES> for their contributions.
- This project was created by <YOUR_NAME>. Find me on [GitHub](https://github.com/<YOUR_GITHUB_USERNAME>) and [LinkedIn](https://www.linkedin.com/in/<YOUR_LINKEDIN_USERNAME>/).
Copy the text above into your README.md file and replace the placeholders with your project-specific information. This will give your users an overview of your project, how to install and use it, and how they can contribute.







