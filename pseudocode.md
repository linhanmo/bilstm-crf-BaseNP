# BaseNP Pseudocode

This document presents concise pseudocode for the current BaseNP pipeline, including data preparation, `jieba` baseline inference, and the main sequence labeling models.

## 1. CTB To BaseNP Data Preparation

```text
Algorithm BUILD_BASENP_DATASET(ctb_parse_files, output_dir)
Input:
    ctb_parse_files = {train, dev, test}
    output_dir
Output:
    BaseNP TSV files

Begin
    for each split_file in ctb_parse_files do
        for each parse_tree in split_file do
            tokens <- EXTRACT_TERMINALS(parse_tree)
            spans <- EXTRACT_MAXIMAL_ACCEPTABLE_NP_SPANS(parse_tree)
            tags <- CONVERT_SPANS_TO_BIOES(tokens, spans, label = "NP")
            WRITE (token, tag) pairs to output file
            WRITE blank line after each sentence
        end for
    end for
End
```

```text
Algorithm EXTRACT_MAXIMAL_ACCEPTABLE_NP_SPANS(tree)
Input:
    tree
Output:
    base_np_spans

Begin
    TRAVERSE tree bottom-up
    for each NP node do
        if node contains forbidden phrase labels then
            reject node
        else if node contains DEC, DEG, DEV, or DER then
            reject node
        else if node is temporal/function-like NP then
            reject node
        else if node is a valid larger NP then
            keep parent NP and drop smaller nested NP spans
        end if
    end for
    return accepted spans
End
```

## 2. Jieba Baseline For BaseNP

```text
Algorithm JIEBA_BASENP(sentence_tokens)
Input:
    sentence_tokens
Output:
    basenp_tags

Begin
    raw_text <- CONCAT(sentence_tokens)
    segmented_items <- JIEBA_POSSEG(raw_text)
    np_spans <- []

    for each item in segmented_items do
        if POS is noun-like or modifier-like then
            EXPAND current NP candidate
        else
            CLOSE current NP candidate if it has a noun head
        end if
    end for

    ALIGN np_spans back to dataset tokens
    basenp_tags <- CONVERT_SPANS_TO_BIOES(sentence_tokens, np_spans, label = "NP")
    return basenp_tags
End
```

## 3. CRF Pseudocode

```text
Algorithm CRF_TRAIN(train_sentences, train_tags)
Input:
    train_sentences, train_tags
Output:
    trained_crf_model

Begin
    for each (sentence, tags) in training data do
        features <- EXTRACT_TOKEN_FEATURES(sentence)
        ADD (features, tags) to training_set
    end for

    INITIALIZE CRF parameters
    OPTIMIZE conditional log-likelihood
    return trained_crf_model
End
```

```text
Algorithm CRF_PREDICT(sentence, trained_crf_model)
Input:
    sentence, trained_crf_model
Output:
    basenp_tags

Begin
    features <- EXTRACT_TOKEN_FEATURES(sentence)
    basenp_tags <- VITERBI_DECODE(features, trained_crf_model)
    return basenp_tags
End
```

## 4. HMM Pseudocode

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
    return hmm_model
End
```

```text
Algorithm HMM_PREDICT(sentence, hmm_model, states)
Input:
    sentence, hmm_model, states
Output:
    basenp_tags

Begin
    basenp_tags <- HMM_VITERBI_DECODE(sentence, hmm_model, states)
    return basenp_tags
End
```

## 5. BiLSTM Pseudocode

```text
Algorithm BILSTM_TRAIN(train_sentences, train_tags)
Input:
    train_sentences, train_tags
Output:
    trained_bilstm_model

Begin
    BUILD vocabulary and tag mappings
    CONVERT sentences and tags into ids

    for each epoch do
        for each batch do
            embeddings <- LOOKUP(batch_tokens)
            contextual_repr <- BiLSTM(embeddings)
            logits <- LINEAR(contextual_repr)
            loss <- TOKEN_CLASSIFICATION_LOSS(logits, gold_tags)
            BACKPROPAGATE and UPDATE parameters
        end for
    end for

    return trained_bilstm_model
End
```

## 6. BiLSTM-CRF Pseudocode

```text
Algorithm BILSTM_CRF_TRAIN(train_sentences, train_tags)
Input:
    train_sentences, train_tags
Output:
    trained_bilstm_crf_model

Begin
    BUILD vocabulary and tag mappings
    CONVERT sentences and tags into ids

    for each epoch do
        for each batch do
            embeddings <- LOOKUP(batch_tokens)
            contextual_repr <- BiLSTM(embeddings)
            emission_scores <- LINEAR(contextual_repr)
            loss <- CRF_NEGATIVE_LOG_LIKELIHOOD(emission_scores, gold_tags)
            BACKPROPAGATE and UPDATE parameters
        end for
    end for

    return trained_bilstm_crf_model
End
```

## 7. End-To-End Pipeline

```text
Algorithm RUN_PIPELINE()
Input:
    CTB archive or preprocessed CTB files
Output:
    trained models, reports, comparison results

Begin
    if BaseNP dataset does not exist or rebuild is requested then
        PREPARE CTB files if necessary
        BUILD_BASENP_DATASET(...)
    end if

    TRAIN HMM
    TRAIN CRF
    TRAIN BiLSTM
    TRAIN BiLSTM-CRF
    RUN model comparison with jieba baseline
    SAVE all outputs under outputs/
End
```

## 8. Short Comparison

```text
Jieba baseline:
    POS-guided heuristic NP extraction
    No supervised training

HMM:
    Generative sequence labeling
    Uses transition and emission probabilities

CRF:
    Discriminative sequence labeling
    Uses handcrafted contextual features

BiLSTM:
    Neural token classification with contextual encoding

BiLSTM-CRF:
    BiLSTM encoder + CRF decoding
    Stronger sequence-level label consistency
```
