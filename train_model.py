# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:28:43 2019

@author: Brolof
"""

import time
import numpy
import sys
import torch
import torch.nn as nn
from data.pytorch_dataset_hdf5_wav import Dataset, AudioDataLoader
from cnn_model import ConvNet2 as Net
from data.import_data_GSC import load_data_set_indexes
from early_stopping import EarlyStopping
from ctc_decoder import decode_sample, compute_edit_distance
from analytics.logger import TensorboardLogger, VisdomLogger


# noinspection PyUnresolvedReferences
def print_cuda_information(using_cuda):
    print("############ CUDA Information #############")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    if using_cuda:
        print(' Using CUDA ')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Name of CUDA Devices:', torch.cuda.get_device_name(device))
        # print('__Devices')
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print('Cuda capability of device: ', torch.cuda.get_device_capability(device=torch.cuda.current_device()))
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
    else:
        print(' Not using CUDA ')
    print("#######################################")


def create_dataloaders(hdf5file_path_in, partition_in, params_in):
    # Generators
    training_set = Dataset(hdf5file_path_in, partition_in['train'])
    training_generator_out = AudioDataLoader(training_set, **params_in)

    validation_set = Dataset(hdf5file_path_in, partition_in['validation'])
    validation_generator_out = AudioDataLoader(validation_set, **params_in)

    testing_set = Dataset(hdf5file_path_in, partition_in['test'])
    testing_generator_out = AudioDataLoader(testing_set, **params_in)
    return training_generator_out, validation_generator_out, testing_generator_out


def print_metrics(current_epoch, current_batch, total_batches, loss, edit_distance, start_time, max_epochs, batch_size):
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Edit Distance: {:.2f}, Time: {:.2f}s, Sample/s: {:.2f}'
          .format(current_epoch + 1, max_epochs, current_batch + 1, total_batches, loss.item(),
                  edit_distance, (time.time() - start_time),
                  batch_size * print_frequency / (time.time() - start_time)))


def train_model(model_input, training_generator, validation_generator, max_epochs, batch_size, optimizer, criterion, using_cuda):
    """

    :param model_input:
    :param training_generator:
    :param validation_generator:
    :param max_epochs:
    :param batch_size:
    :param optimizer:
    :param criterion:
    :param using_cuda:
    """
    # temp_time = time.time()
    print('Function train_model called by: ', repr(__name__))
    total_time = time.time()
    model = model_input
    exit_early = False
    train_losses = []
    valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    n_training_batches = len(training_generator)
    n_validation_batches = len(validation_generator)
    print("Starting training:")
    batch_time = time.time()
    # Todo: Proper tracking, like tensorboard
    for epoch in range(max_epochs):

        # Training
        print("Epoch ", epoch, "/", max_epochs, " starting.")
        for i, (local_data) in enumerate(training_generator, 0):
            local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
            input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()

            if epoch == 0 & i == 0:
                logger_args = [local_batch, input_lengths]
                # logger.add_model_graph(model, logger_args)

            # Transfer to GPU
            if using_cuda:  # On GPU
                local_batch, local_targets = local_batch.cuda(non_blocking=True), local_targets.cuda(non_blocking=True)
            # Run the forward pass
            outputs, output_lengths = model(local_batch, input_lengths)
            # Compute loss
            loss = criterion(outputs, local_targets, output_lengths, local_target_lengths)
            batch_loss = loss.item()

            train_losses.append(batch_loss)

            # Backpropagation and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_frequency == 0:
                evaluated_label = decode_sample(outputs, 0)
                true_label = local_targets[0, :local_target_lengths[0]]
                edit_distance = compute_edit_distance(outputs, local_targets, local_target_lengths, 0)
                print('Evaluated: ', evaluated_label)
                print('True:      ', true_label)

                logger.update_scalar('continuous/loss', batch_loss)
                visdom_logger.update(batch_loss, edit_distance)

                print_metrics(current_epoch=epoch, current_batch=i, total_batches=n_training_batches, loss=loss,
                              edit_distance=edit_distance, start_time=batch_time, max_epochs=max_epochs, batch_size=batch_size)
                batch_time = time.time()

            if ((i == maxNumBatches) & endEarlyForProfiling):
                print(" Exiting program early! ")
                exit_early = True
                break

        # Validation
        if not exit_early:
            with torch.set_grad_enabled(False):
                model.eval()
                batch_time = time.time()
                for (local_data) in validation_generator:
                    local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
                    input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()
                    # Transfer to GPU
                    if using_cuda:  # On GPU
                        local_batch, local_targets = local_batch.cuda(non_blocking=True), local_targets.cuda(
                            non_blocking=True)
                    # Run the forward pass
                    outputs, output_lengths = model(local_batch, input_lengths)
                    loss = criterion(outputs, local_targets, output_lengths, local_target_lengths)  # CTC loss function

                    valid_losses.append(loss.item())

                # print training/validation statistics
                # calculate average loss over an epoch
                #train_loss = numpy.average(train_losses)
                valid_loss = numpy.average(valid_losses)
                train_losses = []
                valid_losses = []
                early_stopping(valid_loss, model)

            print('---------------------------------------------------------------------------------------------')
            per = 0
            print_metrics(current_epoch=epoch, current_batch=n_validation_batches-1, total_batches=n_validation_batches,
                          loss=loss, edit_distance=edit_distance, start_time=batch_time, max_epochs=max_epochs, batch_size=batch_size)
            print('#############################################################################################')
            if early_stopping.early_stop:
                print("Early stopping")
                break
            model.train()


    print('Finished Training')
    print("Total training time: ", (time.time() - total_time), " s")


def evaluate_on_testing_set(model_in, testing_generator_in, criterion):
    import math
    # Testing
    testing_losses = []
    testing_edit_distances = []
    n_testing_batches = len(testing_generator_in)
    with torch.set_grad_enabled(False):
        correct = 0
        total = 0
        i = 0
        model_in.eval()
        for local_data in testing_generator_in:
            print("Evaluating on test data: [{}/{}]".format(i, n_testing_batches))
            i += 1
            # Transfer to GPU
            local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
            input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()

            # Transfer to GPU
            if use_cuda:  # On GPU
                local_batch, local_targets = local_batch.cuda(non_blocking=True), local_targets.cuda(
                    non_blocking=True)

            # Track the accuracy
            outputs, output_lengths = model_in(local_batch, input_lengths)
            loss = criterion(outputs, local_targets, output_lengths, local_target_lengths)  # CTC loss function
            batch_size = local_targets.size(0)
            total += batch_size
            batch_size = math.floor(batch_size*0.25)
            testing_losses.append(loss.item())
            edit_distance = compute_edit_distance(outputs, local_targets, local_target_lengths, batch_size)
            testing_edit_distances.append(edit_distance)

            testing_loss = numpy.average(testing_losses)
            testing_edit_distance = numpy.average(testing_edit_distances)

            print('Testing Accuracy of the model on the ', total,
                  ' testing images. Loss: {:.3f}, PER: {:.4f}'.format(testing_loss, testing_edit_distance))


def visualize_data_from_loader(training_generator_in, validation_generator_in):
    print('-------------------------------------------------------------')
    print('--------- Training Generator --------------------------------')
    for i, (local_batch, local_labels) in enumerate(training_generator_in, 0):
        print(numpy.unique(local_labels))
    print('-------------------------------------------------------------')
    print('--------- Validation Generator --------------------------------')
    for i, (local_batch, local_labels) in enumerate(validation_generator_in, 0):
        print(numpy.unique(local_labels))


if __name__ == "__main__":
    # Parameters
    continue_training = True
    model_path_to_train = "./trained_models/checkpoint.pt"
    model_path_to_evaluate = "./trained_models/checkpoint.pt"  # checkpoint.pt  CNN-BLSTMx2 = 0.1176 PER
    endEarlyForProfiling = False
    maxNumBatches = 101
    runOnCPUOnly = False
    if endEarlyForProfiling:
        max_epochs_training = 1
    else:
        max_epochs_training = 500

    mini_batch_size = 400
    print_frequency = 20
    patience = 3
    learning_rate = 1e-3 # 1e-3 looks good, 1e-2 is too high

    # Datasets test
    dataset_hdf5_path = "../data/GoogleSpeechCommands/hdf5_format/"
    hdf5file_path = dataset_hdf5_path + "allWavIdx.hdf5"

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available() & (not runOnCPUOnly)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not use_cuda:
        numberOfWorkers = 0
        pin_memory = False
    else:
        numberOfWorkers = 4
        pin_memory = True

    params = {'batch_size': mini_batch_size,
              'shuffle': True,
              'num_workers': numberOfWorkers,
              'pin_memory': pin_memory}

    print_cuda_information(use_cuda)

    model_to_train = Net()  # ConvNet()
    partition = load_data_set_indexes(dataset_hdf5_path)
    training_dataloader, validation_dataloader, testing_dataloader = create_dataloaders(hdf5file_path, partition, params)

    logger = TensorboardLogger()
    visdom_logger = VisdomLogger("Loss", 20)

    criterion_ctc = nn.CTCLoss(zero_infinity=True)
    optimizer_adam = torch.optim.Adam(model_to_train.parameters(), lr=learning_rate)
    if use_cuda:
        model_to_train.cuda()
        # optimizer.cuda()

    if continue_training:
        model_to_train.load_state_dict(torch.load(model_path_to_train))

    print("--------Calling train_model()")
    train_model(model_to_train, training_dataloader, validation_dataloader, max_epochs_training, mini_batch_size,
                optimizer_adam, criterion_ctc, use_cuda)
    if not endEarlyForProfiling:
        model_to_evaluate = model_to_train
        model_to_evaluate.load_state_dict(torch.load(model_path_to_evaluate))
        evaluate_on_testing_set(model_to_evaluate, testing_dataloader, criterion_ctc)
    if use_cuda:
        print('Maximum GPU memory occupied by tensors:', torch.cuda.max_memory_allocated(device=None)/1e9, 'GB')
        print('Maximum GPU memory managed by the caching allocator: ',
              torch.cuda.max_memory_cached(device=None)/1e9, 'GB')
