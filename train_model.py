# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:28:43 2019

@author: Brolof
"""

import time
import numpy
import sys
import torch
from torch.utils import data
import torch.nn as nn
from pytorch_dataset_hdf5 import Dataset
from Scripts.import_data import ImportData
from cnn_model import ConvNet
from torch.autograd import Variable
from csv_to_list import csvToList
from early_stopping import EarlyStopping


def print_cuda_information(use_cuda):
    print('Script v1.1')
    print("############ CUDA Information #############")
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    if use_cuda:
        print(' Using CUDA ')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Name of CUDA Devices:', torch.cuda.get_device_name(device))
        # print('__Devices')
        # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
    else:
        print(' Not using CUDA ')
    print("#######################################")


def load_data_set_path(dataset_path):
    # VERBOSE = False
    partition, labels, label_index_ID_table = ImportData.importData(dataset_path,35)# IDs
    # labels = # Labels
    return partition, labels


def load_data_set_indexes(dataset_path_in):
    partition_out = {'train': csvToList(dataset_path_in + "train_idx.csv"),
                 'validation': csvToList(dataset_path_in + "validation_idx.csv"),
                 'test': csvToList(dataset_path_in + "test_idx.csv")}

    return partition_out


def create_dataloaders(hdf5file_path_in, partition_in, params_in):
    # Generators
    training_set = Dataset(hdf5file_path_in, partition_in['train'])
    training_generator_out = data.DataLoader(training_set, **params_in)

    validation_set = Dataset(hdf5file_path_in, partition_in['validation'])
    validation_generator_out = data.DataLoader(validation_set, **params_in)

    testing_set = Dataset(hdf5file_path_in, partition_in['test'])
    testing_generator_out = data.DataLoader(testing_set, **params_in)
    return training_generator_out, validation_generator_out, testing_generator_out


def print_metrics(current_epoch, current_batch, total_batches, loss, n_correct_pred, n_total_samples, start_time):
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Time: {:.2f}s, Sample/s: {:.2f}'
          .format(current_epoch + 1, max_epochs, current_batch + 1, total_batches, loss.item(),
                  (n_correct_pred / n_total_samples) * 100, (time.time() - start_time),
                  batch_size * print_frequency / (time.time() - start_time)))


def trainModel(model_input, training_generator, validation_generator, max_epochs, optimizer, criterion, use_cuda):
    # temp_time = time.time()
    print('Function trainModel called by: ', repr(__name__))
    total_time = time.time()
    model = model_input

    train_loss = []
    train_losses = []
    valid_loss = []
    valid_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    acc_list = []
    acc_list_val = []
    n_training_batches = len(training_generator)
    n_validation_batches = len(validation_generator)
    print("Starting training:")
    batch_time = time.time()
    for epoch in range(max_epochs):
        # Training
        print("Epoch ", epoch, "/", max_epochs, " starting.")
        for i, (local_batch, local_labels) in enumerate(training_generator, 0):

            # Transfer to GPU
            if use_cuda:  # On GPU
                local_batch, local_labels = local_batch.cuda(non_blocking=True), local_labels.cuda(non_blocking=True)
            else:
                local_batch, local_labels = Variable(local_batch), Variable(
                    local_labels)  # RuntimeError: expected CPU tensor (got CUDA tensor)

            # Run the forward pass
            outputs = model(local_batch)
            if use_cuda:
                loss = criterion(outputs, local_labels).cuda()
            else:
                loss = criterion(outputs, local_labels)
            train_losses.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = local_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)

            # torch.cuda.synchronize()
            correct = (predicted == local_labels).sum().item()
            # <- This requires a synchronize of cuda which takes a lot of time. Solution?
            acc_list.append(correct / total)

            if (i + 1) % print_frequency == 0:
                print_metrics(current_epoch=epoch, current_batch=i, total_batches=n_training_batches, loss=loss,
                              n_correct_pred=correct, n_total_samples=total, start_time=batch_time)
                batch_time = time.time()
            # print("Mini batch time: ", (time.time()-single_mini_batch_time)*1000, " ms")
            # single_mini_batch_time = time.time()

            if ((i == maxNumBatches) & endEarlyForProfiling):
                print(" Exiting program early! ")
                print("Total training time: ", (time.time() - total_time), " s")
                exit()
        # Validation
        with torch.set_grad_enabled(False):
            correct = 0
            total = 0
            model.eval()
            batch_time = time.time()
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Track the accuracy
                outputs = model(local_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()

                loss = criterion(outputs, local_labels)
                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = numpy.average(train_losses)
            valid_loss = numpy.average(valid_losses)
            train_losses = []
            valid_losses = []

            early_stopping(valid_loss, model)

        print('---------------------------------------------------------------------------------------------')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print_metrics(current_epoch=epoch, current_batch=n_validation_batches-1, total_batches=totalValidationBatches, loss=loss,
        n_correct_pred=correct, n_total_samples=total, start_time=batch_time)
        print('#############################################################################################')
        model.train()

    model.load_state_dict(torch.load('checkpoint.pt'))
    print('Finished Training')
    print("Total training time: ", (time.time() - total_time), " s")


def evaluateOnTestingSet(model, testing_generator):
    # Validation
    with torch.set_grad_enabled(False):
        correct = 0
        total = 0
        model.eval()
        for local_batch, local_labels in testing_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Track the accuracy
            outputs = model(local_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += local_labels.size(0)
            correct += (predicted == local_labels).sum().item()

    print('Testing Accuracy of the model on the ', total,
          ' testing images: {} %'.format((correct / total) * 100))
# TODO: Save the model and plot


def visualizeDataFromLoader(training_generator, validation_generator):
    print('-------------------------------------------------------------')
    print('--------- Training Generator --------------------------------')
    for i, (local_batch, local_labels) in enumerate(training_generator, 0):
        print(numpy.unique(local_labels))
    print('-------------------------------------------------------------')
    print('--------- Validation Generator --------------------------------')
    for i, (local_batch, local_labels) in enumerate(validation_generator, 0):
        print(numpy.unique(local_labels))
# TODO: Clean up

if __name__ == "__main__":
    # Parameters
    endEarlyForProfiling = False
    maxNumBatches = 21
    runOnCPUOnly = False

    batch_size = 400
    print_frequency = 20
    patience = 3
    max_epochs = 10
    learning_rate = 5e-4

    # Datasets
    dataset_path = "./data/"
    datasethdf5_path = "./input_data_hdf5_70m/"
    hdf5file_path = datasethdf5_path + "testArraySize11.hdf5" #"fileArray.hdf5" #

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available() & (not runOnCPUOnly)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not use_cuda:
        numberOfWorkers = 0
        pin_memory = False
    else:
        numberOfWorkers = 4
        pin_memory = True
    # Hyperparameters
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': numberOfWorkers,
              'pin_memory': pin_memory}

    print_cuda_information(use_cuda)
    model = ConvNet()
    model.share_memory()
    #partition, labels = load_data_set_indexes(dataset_path)
    partition = load_data_set_indexes(datasethdf5_path)
    training_generator, validation_generator, testing_generator = create_dataloaders(hdf5file_path, partition, params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model.cuda()
        # optimizer.cuda()

    print("--------Calling trainModel()")
    mainCallCounter = 0
    trainModel(model, training_generator, validation_generator, max_epochs, optimizer, criterion, use_cuda)
    #visualizeDataFromLoader(training_generator, validation_generator)
    evaluateOnTestingSet(model, testing_generator)
#else:
    # print('New worker started', repr(__name__))


    #print(torch.utils.data.get_worker_info())

# Useful commands:
# torch.save(model.state_dict(), "./models/" + 'conv_net_model.ckpt')
