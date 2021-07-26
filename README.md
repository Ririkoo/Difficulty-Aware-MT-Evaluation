# Difficulty-Aware Machine Translation Evaluation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic Evaluation Metric described in the paper [Difficulty-Aware Machine Translation Evaluation](https://aclanthology.org/2021.acl-short.5/) (ACL 2021).

### Citation
```bibtex
@inproceedings{zhan-etal-2021-difficulty,
    title = "Difficulty-Aware Machine Translation Evaluation",
    author = "Zhan, Runzhe and Liu, Xuebo and Wong, Derek F. and Chao, Lidia S.",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "26--32"
 }
```

### Acknowledgement
This repo wouldn't be possible without the awesome [BERTScore](https://github.com/Tiiiger/bert_score), [bert](https://github.com/google-research/bert), [fairseq](https://github.com/pytorch/fairseq), and [transformers](https://github.com/huggingface/transformers).

### Installation
* Python version >= 3.6
* PyTorch version >= 1.0.0

Install it from the source by:
```sh
git clone https://github.com/NLP2CT/Difficulty-Aware-MT-Evaluation
cd Difficulty-Aware-MT-Evaluation
pip install --editable .
```

### Usage
This package not only preserves the original features of BERTScore (ver.0.3.7) but also can coexist with BERTScore.

#### Parameters
| Parameters  | Descriptions                                                 |
| :---------: | :----------------------------------------------------------- |
| --with_diff | Score multiple MT systems using difficulty information. Otherwise, switch to the original BERTScore implementation. |
| --cand_list | Paths of system output files.                                |
| --save_path | Path of result file                                          |

Please refer to [BERTScore](https://github.com/Tiiiger/bert_score) for other parameters.


#### CLI Example
For reproducing the the WMT19 En-De Top-6 scoring results, you can use it as follows:

```sh
WMT19_DATA_PATH=example_data/wmt19-ende

da-bert-score --with_diff --batch_size 256 --lang de --ref ${WMT19_DATA_PATH}/ref/newstest2019-ende-ref.de \
    --cand_list ${WMT19_DATA_PATH}/sys/en-de/newstest2019.Facebook_FAIR.6862.en-de ${WMT19_DATA_PATH}/sys/en-de/newstest2019.Microsoft-WMT19-sentence_document.6974.en-de ${WMT19_DATA_PATH}/sys/en-de/newstest2019.Microsoft-WMT19-document-level.6808.en-de ${WMT19_DATA_PATH}/sys/en-de/newstest2019.MSRA.MADL.6926.en-de ${WMT19_DATA_PATH}/sys/en-de/newstest2019.UCAM.6731.en-de ${WMT19_DATA_PATH}/sys/en-de/newstest2019.NEU.6763.en-de \
    --save_path wmt2019_res/en-de/topk
```
This implementation follows the [default behaviors (models, layers)](https://github.com/Tiiiger/bert_score#default-behavior) of BERTScore when evaluating the different languages.

#### Variant
We will keep exploring the possible variants of DA-BERTScore.

The default parameter settings are used to reproduce the results reported in the paper. For achieving better correlation results across multiple languages, you can try one variant by enabling the parameter `--ref_diff --softmax_norm --range_one`. The explanations are as follows:

|   Parameters   | Descriptions                                                 |
| :------------: | :----------------------------------------------------------- |
|   --ref_diff   | The weight of the hypothesis word could be unknown if there is no identical word in the reference. By enabling this parameter, it will use the weight of corresponding reference word whose similarity score is maximal. |
| --softmax_ norm | Use the softmax function to smooth the distribution of difficulty weight. |
|  --range_one   | Scale the score to range [0,1]                               |
