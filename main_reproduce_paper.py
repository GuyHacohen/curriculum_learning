#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:53:30 2019

@author: guy.hacohen
"""

from argparse import Namespace
from main_train_networks import run_expriment


def curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.035,
                     lr_decay_rate=1.5,
                     minimal_lr=1e-4,
                     lr_batch_size=300,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

    
def vanilla_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.05,
                     lr_decay_rate=1.8,
                     minimal_lr=1e-4,
                     lr_batch_size=500,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def anti_curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=200,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def random_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.3,
                     minimal_lr=1e-4,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)
    

if __name__ == "__main__":
    output_path = ""
    num_repeats = 7
    
    curriculum_small_mammals(num_repeats, output_path=output_path)
#    vanilla_small_mammals(num_repeats, output_path=output_path)
#    anti_curriculum_small_mammals(num_repeats, output_path=output_path)
#    random_small_mammals(num_repeats, output_path=output_path)
    
    