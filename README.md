# ADRM
The Tensorflow implementation of our TOIS paper:    
***An Attention-based Deep Relevance Model for Few-Shot Document Filtering, Bulou Liu, Chenliang Li, Wei Zhou, Feng Ji, Yu Duan, Haiqing Chen***

Paper url: https://dl.acm.org/doi/10.1145/3419972

### Requirements
- Python 3.5
- Tensorflow 1.2
- Numpy
- Traitlets

### Guide To Use

**Prepare your dataset**: first, prepare your own data.
See [Data Preparation](#data-preparation)

**Training** 

```ruby
train_file: path to training data\
validation_file: path to validation data\
checkpoint_dir: directory to store/load model checkpoints\ 
load_model: True or False(depends on existing or not). Start with a new model or continue training
```

**Testing**

```ruby
test_file: path to testing data\
test_size: size of testing data (number of testing samples)\
checkpoint_dir: directory to load trained model\
output_score_file: file to output documents score\
```
Relevance scores will be output to output_score_file, one score per line, in the same order as test_file.


### Data Preparation


All seed words and documents must be mapped into sequences of integer term ids. Term id starts with 1. 

**Training Data Format**

Each training sample is a tuple of (seed words, postive document, negative document)

`seed_words   \t postive_document   \t negative_document `

Example: `334,453,768   \t  123,435,657,878,6,556   \t  443,554,534,3,67,8,12,2,7,9 `


**Testing Data Format**

Each testing sample is a tuple of (seed words, document)

`seed_words   \t document`

Example: `334,453,768  \t   123,435,657,878,6,556`


**Validation Data Format**

The format is same as training data format


**Label Dict File Format**

Each line is a tuple of (label_name, seed_words)

`label_name/seed_words`

Example: `alt.atheism/atheist christian atheism god islamic`


**Word2id File Format**

Each line is a tuple of (word, id)

`word id`

Example: `world 123`


**Embedding File Format**

Each line is a tuple of (id, embedding)

`id embedding`

Example: `1 0.3 0.4 0.5 0.6 -0.4 -0.2`
