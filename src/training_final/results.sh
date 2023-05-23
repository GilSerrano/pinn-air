#!/usr/bin/env bash

# This script is used to generate the results for the final report.
python train_model.py --dropout 0.0 --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.0 --output_path output_7
python train_model.py --dropout 0.0 --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.05 --output_path output_8
python train_model.py --dropout 0.0 --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.02 --output_path output_9
python train_model.py --dropout 0.0 --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.01 --output_path output_10


python train_model.py --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.10 --dropout 0.0 --output_path ./output_11
python train_model.py --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.07 --dropout 0.0 --output_path ./output_12

python train_model.py --dropout 0.0 --data_augmentation --output_path output_13

python train_model.py --dropout 0.0 --batch_size 16 --num_epochs 1000 --learning_rate 1E-5 --data_augmentation --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.10 --physics_error 1.0 --output_path output_18
python train_model.py --dropout 0.0 --batch_size 16 --learning_rate 1E-5 --teacher_forcing_ratio 0.5 --teacher_forcing_decay 0.10 --physics_error 1.0 --output_path output_19