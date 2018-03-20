#!/bin/bash

# Notes:
# * Remember that the API talks thrift to the backend server

pip install scikit-learn==0.19.1

cd ./samples/scikit && python GridSearchCrossValidation.py
