## Intro

This repository contains exploration into various ways of algorithmically solving the New York Time connections puzzle. 
The puzzle presents the player with 16 words. The players aim is to group the words into clusters of four having something in common. 

## Setup

First, set up a conda environment containing all necessary python libraries for running the code. It can be found in the `environment.yml` file.

```conda env create -f environment.yml```

Next, run 

```git clone https://github.com/Eyefyre/NYT-Connections-Answers.git```

from within the repository to download the archive of all NYT Connections games. 

Ultimately, create a new directory `embeddings` and download desired word embeddings into it. For the purposes of the demo Jupyter notebook `connections.ipynb` download [glove6B.zip](https://nlp.stanford.edu/projects/glove/).

