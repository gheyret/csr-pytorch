# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:25:56 2019

"""

import numpy
import torch


class EarlyStopping:
    """
    Class used to determine when to end training or when to exit program
    """

    def __init__(self, end_early, max_num_batches, delta=0.0, verbose=False, patience=3,
                 checkpoint_path='./trained_models/checkpoint.pt'):
        self.verbose = verbose
        self.delta = delta

        self.end_early = end_early
        self.max_num_batches = max_num_batches
        self.end_early_counter = 0
        self.stop_program = False

        self.stop_training_early = False
        self.patience = patience
        self.lowest_loss = None
        self.lowest_loss_prev = numpy.inf
        self.validation_counter = 0
        self.checkpoint_path = checkpoint_path

    def __call__(self, loss, model):
        """
        This is called at each validation step to determine if the model has stopped learning on validation data.
        :param loss: The validation loss. Lower = better
        :param model: The model that should be saved.
        :return:
        """
        if self.lowest_loss is None:  # First time
            self.lowest_loss = loss
            self.save_model(model)

        elif self.lowest_loss - self.delta > loss:  # loss is lower than previous best
            self.lowest_loss = loss
            self.save_model(model)
            self.validation_counter = 0

        else:  # loss is not low enough
            self.validation_counter += 1
            if self.verbose:
                print("Validation loss didn't improve. Best: {:.3f}, Current: {:.3f} (Delta: {:.4f}). Exiting early: ({}/{})"
                      .format(self.lowest_loss, loss, self.delta, self.validation_counter, self.patience))
            if self.validation_counter >= self.patience:
                self.stop_training_early = True
    def reset(self):
        self.end_early_counter = 0
        self.stop_program = False
        self.stop_training_early = False
        self.lowest_loss = None
        self.lowest_loss_prev = numpy.inf
        self.validation_counter = 0

    def save_model(self, model):
        """
        Saves model to the specified path
        :param model: The model that should be saved.
        :return:
        """
        torch.save(model.state_dict(), self.checkpoint_path)
        if self.verbose:
            print("Validation loss decreased from {:.5f} -> {:.5f}. Saving model at path: '{path}'"
                  .format(self.lowest_loss_prev, self.lowest_loss, path=self.checkpoint_path))
        self.lowest_loss_prev = self.lowest_loss

    def exit_program_early(self):
        """
        Used to determine when to exit the program early.
        This can be used for example to run a profiling faster, or to partially train a model.
        :return:
        """
        if self.end_early:
            self.end_early_counter += 1
            if self.end_early_counter >= self.max_num_batches:
                if self.verbose:
                    print(" Exiting program early! ")
                self.stop_program = True
