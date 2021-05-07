WaveNet Autoencoder with Contrastive Predictive Coding for Music Translation
==============================

WaveNet autoencoder using Contrastive Predictive Coding for music translation with raw audio

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── samples            <- Generated audio samples
    │   ├── Beethoven_Solo_Piano  <- Generated samples from Beethoven solo piano input samples
    │   │   │
    │   │   └── 0.wav        <- Original sample #0
    │   │   │
    │   │   └── umt_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using UMT
    │   │   │
    │   │   └── umt_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using UMT
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │   │
    │   ├── Solo_Cello     <- Generated samples from solo cello input samples
    │   │   │
    │   │   └── 0.wav        <- Original sample #0
    │   │   │
    │   │   └── umt_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using UMT
    │   │   │
    │   │   └── umt_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using UMT
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │   │
    │   ├── Solo_Violin    <- Generated samples from solo cello input samples
    │   │   │
    │   │   └── 0.wav        <- Original sample #0
    │   │   │
    │   │   └── umt_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using UMT
    │   │   │
    │   │   └── umt_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using UMT
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_0.wav  <- Sample #0 translated into domain 0 (Solo_Cello) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_1.wav  <- Sample #0 translated into domain 1 (Solo_Violin) using CPC WaveNet with GRU autoregressor
    │   │   │
    │   │   └── umtcpc-gru_0_2.wav  <- Sample #0 translated into domain 2 (Beethoven_Solo_Piano) using CPC WaveNet with GRU autoregressor
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── shell              <- Shell scripts that invoke source code
    |
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   └── data           <- Scripts to download or generate data
    │   │
    │   └── evaluation     <- Scripts to evaluate trained models
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
