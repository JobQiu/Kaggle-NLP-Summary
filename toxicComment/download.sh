#!/usr/bin/env bash

DATA_DIR=/content/data
OUT_DIR=/content/out/
mkdir $DATA_DIR
mkdir $OUT_DIR
pip install kaggle

rm /root/.kaggle/kaggle.json

# download the API credentials
wget https://raw.githubusercontent.com/JobQiu/EloMerchantKaggle/master/data/kaggle.json -P /root/.kaggle/
chmod 600 /root/.kaggle/kaggle.json
#


if [ -e /content/data/sample_submission.csv ]
then
    echo "Files have been extracted already."

else

    if [ -e /content/data/embeddings.zip ]
    then
        echo "Files have been downloaded already."
    else
        kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p /content/data
    fi

    unzip /content/data/train.csv.zip -d /content/data
    unzip /content/data/test.csv.zip -d /content/data
    unzip /content/data/sample_submission.csv.zip -d /content/data

fi
