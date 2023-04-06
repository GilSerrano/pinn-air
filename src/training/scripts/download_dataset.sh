#!/usr/bin/env bash

mkdir -p dataset/sim_circles
cd dataset/sim_circles

# Download the simulated circle trajectory dataset
wget -O dataset.zip 

# Unzip the dataset
unzip dataset.zip

# Remove the zip file
rm dataset.zip