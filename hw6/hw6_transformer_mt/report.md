# Transformer Machine Translation Model Report

## Problem
In this homework, I developed a Transformer-based machine translation model to translate sentences from English to Germany. The model architecture consists of an encoder-decoder structure with multi-head attention and feed-forward layers.

## Dataset
I used the WMT14 English-German Translation dataset as my training data. It consists of pairs of English and Germany sentences. The dataset contains various topics and styles.

## Training Efforts
Due to the size of the model, I wasn't able to do hyper parameter tuning. So, I used the default configuation with vocabulary sizes of 32_000 for the source and target tokenizer. To speed up the training, I used the highest possible batch size that I can train on the GPU I used.