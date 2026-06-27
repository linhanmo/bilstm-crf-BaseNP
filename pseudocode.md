# Word Segmentation Pseudocode

This document presents concise pseudocode for the main ideas behind `jieba`, `CRF`, and `HMM`.

## 1. Jieba Pseudocode

```text
Algorithm JIEBA_SEGMENT(sentence, prefix_dict, word_freq, total_freq, hmm_model)
Input:
    sentence, prefix_dict, word_freq, total_freq, hmm_model
Output:
    segmented_words

Begin
    dag <- BUILD_DAG(sentence, prefix_dict, word_freq)
    route <- DYNAMIC_PROGRAMMING(sentence, dag, word_freq, total_freq)
    coarse_words <- RESTORE_PATH(sentence, route)
    segmented_words <- HMM_REFINE_UNKNOWN_WORDS(coarse_words, word_freq, hmm_model)
    return segmented_words
End
```

## 2. CRF Pseudocode

```text
Algorithm CRF_TRAIN(train_sentences, train_tags)
Input:
    train_sentences, train_tags
Output:
    trained_crf_model

Begin
    for each (sentence, tags) in training data do
        features <- EXTRACT_LOCAL_FEATURES(sentence)
        ADD (features, tags) to training_set
    end for

    INITIALIZE parameters
    OPTIMIZE conditional log-likelihood
    return trained_crf_model
End
```

```text
Algorithm CRF_SEGMENT(sentence, trained_crf_model)
Input:
    sentence, trained_crf_model
Output:
    segmented_words

Begin
    features <- EXTRACT_LOCAL_FEATURES(sentence)
    tags <- VITERBI_DECODE(features, trained_crf_model)
    segmented_words <- TAGS_TO_WORDS(sentence, tags)
    return segmented_words
End
```

## 3. HMM Pseudocode

```text
Algorithm HMM_TRAIN(train_sentences, train_tags, states)
Input:
    train_sentences, train_tags, states
Output:
    hmm_model = (Pi, A, B)

Begin
    for each (sentence, tags) in training data do
        COUNT initial, transition, and emission statistics
    end for

    APPLY smoothing
    NORMALIZE Pi, A, and B
    return (Pi, A, B)
End
```

```text
Algorithm HMM_SEGMENT(sentence, hmm_model, states)
Input:
    sentence, hmm_model, states
Output:
    segmented_words

Begin
    tags <- HMM_VITERBI_DECODE(sentence, hmm_model, states)
    segmented_words <- TAGS_TO_WORDS(sentence, tags)
    return segmented_words
End
```

## 4. Short Comparison

```text
Jieba:
    Dictionary + DAG + dynamic programming
    HMM fallback for unknown words

CRF:
    Discriminative sequence labeling
    Uses contextual features

HMM:
    Generative sequence model
    Uses transition and emission probabilities
```
