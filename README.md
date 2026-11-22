# Hindi News Summarization

Automatic summarization of Hindi news articles, comparing four approaches: a TF-IDF extractive baseline, a Transformer built from scratch, and two fine-tuned pretrained models (mT5 and IndicBART). Built for the CSE556 Natural Language Processing course.

**Best result:** fine-tuned IndicBART, with a ROUGE-1 F-score of 0.554 and a BERTScore F1 of 0.797 on the held-out test set.

## Approaches

| Approach | Type | Notebook |
|---|---|---|
| TF-IDF | Extractive. Scores each sentence by the TF-IDF weights of its words and keeps the top sentences. | [`TF-IDF/tf-idf.ipynb`](Codes%20and%20Models/TF-IDF/tf-idf.ipynb) |
| Custom Transformer | Abstractive. An encoder-decoder Transformer written in TensorFlow/Keras and trained from scratch. | [`CustomTransformer/project-transformer-model.ipynb`](Codes%20and%20Models/CustomTransformer/project-transformer-model.ipynb) |
| mT5 | Abstractive. Fine-tuned from `csebuetnlp/mT5_multilingual_XLSum` (12 encoder and 12 decoder layers). | [`MT5/mt5.ipynb`](Codes%20and%20Models/MT5/mt5.ipynb) |
| IndicBART | Abstractive. Fine-tuned from `ai4bharat/IndicBART` (244M parameters, pretrained on Indian languages). | [`IndicBART/IndicBART_finetuned_concat.ipynb`](Codes%20and%20Models/IndicBART/IndicBART_finetuned_concat.ipynb) |

## Data

- **Training:** `hindi_train.csv` from the ILSUM 2023 shared task, about 20,000 Hindi news articles, each with a heading and a reference summary. It is not included here because of its size.
- **Testing:** [`Dataset/test.csv`](Dataset/test.csv), 2,842 articles with reference summaries from ILSUM 2022. [`Dataset/HindiNews_test.csv`](Dataset/HindiNews_test.csv) holds a further 3,000 articles without summaries.

**Pre-processing:** removed special characters and extra spaces, and converted numbers to Hindi words. For IndicBART and the custom Transformer it also removed English characters.

## Results

Scores on the test set, from [`Group5_Report.pdf`](Group5_Report.pdf):

| Metric | IndicBART | mT5 | Custom Transformer |
|---|---|---|---|
| BERTScore Precision | 0.771 | **0.783** | 0.612 |
| BERTScore Recall | **0.826** | 0.765 | 0.542 |
| BERTScore F1 | **0.797** | 0.773 | 0.737 |
| ROUGE-1 Precision | 0.477 | **0.525** | 0.441 |
| ROUGE-1 Recall | **0.684** | 0.444 | 0.369 |
| ROUGE-1 F-score | **0.554** | 0.471 | 0.393 |
| ROUGE-2 Precision | 0.358 | **0.395** | 0.162 |
| ROUGE-2 Recall | **0.562** | 0.316 | 0.149 |
| ROUGE-2 F-score | **0.429** | 0.342 | 0.151 |
| ROUGE-L Precision | 0.435 | **0.477** | 0.356 |
| ROUGE-L Recall | **0.626** | 0.405 | 0.300 |
| ROUGE-L F-score | **0.506** | 0.429 | 0.318 |

IndicBART had the best F-scores overall, mainly because of much higher recall. mT5 had slightly higher precision but lower recall. The custom Transformer, trained from scratch on about 20,000 examples, lagged well behind both pretrained models. The TF-IDF baseline is covered separately in section 7 of the report.

| IndicBART loss | mT5 loss | Custom Transformer loss |
|---|---|---|
| ![IndicBART training and validation loss](NLP_Project_Images/IndicBART_2.jpg) | ![mT5 training and validation loss](NLP_Project_Images/MT5_2.jpg) | ![Custom Transformer loss](NLP_Project_Images/Transformer.jpg) |

## Repository layout

```
Codes and Models/
  TF-IDF/              extractive baseline
  CustomTransformer/   notebook, trained Keras model and weights, tokenizers (pickles)
  MT5/                 notebook and fine-tuning config
  IndicBART/           notebooks for three fine-tuning runs, and model config
Dataset/               test sets
NLP_Project_Images/    loss plots and sample outputs
Group5_Report.pdf      full report: method, results and qualitative analysis
Poster.pdf             project poster
```

The trained custom Transformer is included (`my_model_transformer.keras`, `transformer.weights.h5`). The fine-tuned mT5 and IndicBART weights were too large for GitHub, so only their configuration files are here. Re-running the notebooks reproduces them.

## Running the notebooks

The notebooks were run on Kaggle and Google Colab. The mT5 and IndicBART fine-tuning needs a GPU. Install the main dependencies:

```bash
pip install torch transformers tensorflow pandas rouge rouge-score bert-score nltk num-to-words emoji tqdm
```

Then download `hindi_train.csv` from ILSUM and update the `read_csv` paths at the top of each notebook (they point at `/kaggle/...` or `/content/...`).
