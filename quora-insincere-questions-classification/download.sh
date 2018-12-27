#!/usr/bin/env bash

DATA_DIR=/content/data
mkdir $DATA_DIR
pip install kaggle

rm /root/.kaggle/kaggle.json

# download the API credentials
wget https://raw.githubusercontent.com/JobQiu/EloMerchantKaggle/master/data/kaggle.json -P /root/.kaggle/
chmod 600 /root/.kaggle/kaggle.json
#

if [ -e /content/data/embeddings.zip ]
then
    echo "files have been downloaded"
else
    kaggle competitions download -c quora-insincere-questions-classification -p /content/data
fi


if [ -e /content/data/sample_submission.csv ]
then
    echo "files have been extracted"
else
    unzip /content/data/train.csv.zip -d /content/data
    unzip /content/data/test.csv.zip -d /content/data
    unzip /content/data/sample_submission.csv.zip -d /content/data
    unzip /content/data/embeddings.zip -d /content/data
fi
