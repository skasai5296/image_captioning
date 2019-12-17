# Image captioning with PyTorch (mainly focused on reinforcement learning)

work is still in progress

## Main Requirements
- Python >= 3.7 (Anaconda recommended)
- Java >= 1.6 (for nlg-eval)

## Setup
- Run `conda env create --file environment.yml && conda activate captioning` if you are a conda user
- Run `python -m spacy download en`
- Install `nlg-eval` (instructions [here](https://github.com/Maluuba/nlg-eval/blob/master/README.md))

## TODO
- [x] Implement dataset loading for MSCOCO
- [x] Add basic training code (models, loss)
- [x] Add evaluation (BLEU, METEOR, CIDEr, ...)
