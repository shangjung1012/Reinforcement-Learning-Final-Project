#!/usr/bin/env bash
set -e

# Create directories
mkdir -p natural-questions
mkdir -p HotpotQA
mkdir -p scifact
mkdir -p nfcorpus

#  Downloading Natural Questions dataset
uvx --from huggingface-hub hf download google-research-datasets/natural_questions \
  --repo-type dataset \
  --local-dir ./natural-questions

#  Downloading BEIR SciFact dataset with qrels
wget -c https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
rm -rf scifact
unzip -q scifact.zip

#  Downloading BEIR NFCorpus dataset with qrels
#  Small full-corpus biomedical/nutrition retrieval benchmark.
wget -c https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
rm -rf nfcorpus
unzip -q nfcorpus.zip

#  Downloading HotpotQA dataset
cd HotpotQA

wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

cd ..

echo "All downloads completed!"
