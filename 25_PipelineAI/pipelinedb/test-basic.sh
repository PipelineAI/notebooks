#!/bin/bash

# Notes:
# * You might have to shimmy the syncer.json a bit to match our environment
# * Remember that the API talks thrift to the backend server
# * You may have to fight the thrift install

cd ./samples/python && python BasicWorkflow.py
