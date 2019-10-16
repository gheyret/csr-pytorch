# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:28:43 2019

@author: Brolof
"""
 
import time
import sys
import torch
from torch.utils import data
import torch.nn as nn
from pytorch_dataset import Dataset
from import_data import ImportData
from cnn_model import ConvNet
from torch.autograd import Variable

import torch.multiprocessing as mp



 
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#device = torch.device("cpu")
#cudnn.benchmark = True

# Hyperparameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 1}
max_epochs = 1
num_classes = 10
learning_rate = 0.001


# Datasets
dataset_path = "./data/"
VERBOSE = False
partition, labels, label_index_ID_table = ImportData.importData(dataset_path,35,VERBOSE)# IDs
#labels = # Labels


# Generators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

#def train(model):
#if __name__ == '__main__': 




# Train the model



#if (use_cuda):
#model = ConvNet().cuda() # GPU
#else:
#    model = ConvNet() # CPU

# Loss and optimizer



#optimizer.cuda()

   

# Loop over epochs

def trainModel(model_input): 

    
    

    
    print('Function trainModel called by: ' , repr(__name__))     
    total_time = time.time()
    model = model_input
    if use_cuda:
        model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    loss_list = []
    acc_list = []
    total_step = len(training_generator)
    print("Starting training:")
    start_time = time.time()
    single_mini_batch_time = start_time
    for epoch in range(max_epochs):
        # Training
        print("Epoch ", epoch, "/", max_epochs, " starting.")
        for i, (local_batch, local_labels) in enumerate(training_generator, 0):
            #print("Inside")
            # get the inputs; data is a list of [inputs, labels]
        #for local_batch, local_labels in training_generator:
            
            # Transfer to GPU
            if use_cuda:
                local_batch, local_labels = local_batch.cuda(non_blocking=True), local_labels.cuda(non_blocking=True) # On GPU                
            else:            
                local_batch, local_labels = Variable(local_batch), Variable(local_labels) # RuntimeError: expected CPU tensor (got CUDA tensor)                           
            
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            # Model computations
            
            # Run the forward pass
            #print(local_batch.is_cuda)
            outputs = model(local_batch)
            if use_cuda:
                loss = criterion(outputs, local_labels).cuda()
            else:
                loss = criterion(outputs, local_labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy
            total = local_labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == local_labels).sum().item()
            acc_list.append(correct / total)
    
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, Time: {:.2f}s'
                      .format(epoch + 1, max_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100, (time.time()-start_time)))
                start_time = time.time()
            print("Mini batch time: ", (time.time()-single_mini_batch_time)*1000, " ms")
            single_mini_batch_time = time.time()
            
            if (i == 31):
                break
            
                # Validation
                with torch.set_grad_enabled(False):
                    correct = 0
                    total = 0
                    model.eval()
                    for local_batch, local_labels in validation_generator:
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            
                        # Track the accuracy
                        outputs = model(local_batch)
                        _, predicted = torch.max(outputs.data, 1)
                        total += local_labels.size(0)
                        correct += (predicted == local_labels).sum().item()
                    print('Validation Accuracy of the model on the 9981 validation images: {} %'.format((correct / total) * 100))
                    model.train()
    print('Finished Training')
    print("Total training time: ", (time.time()-total_time), " s")



# Save the model and plot
#torch.save(model.state_dict(), "./models/" + 'conv_net_model.ckpt')

def runMultipleThreads():
    print('Function called by: ' , repr(__name__))
    #if __name__ == '__main__':
    num_processes = 2
    # NOTE: this is required for the ``fork`` method to work
    model = ConvNet()
    model.share_memory()
    processes = []
    for rank in range(num_processes):
       p = mp.Process(target=trainModel, args=(model,))
       p.start()
       processes.append(p)
       for p in processes:
           p.join()
    
    
#if __name__ == '__main__':
#    num_processes = 4
#    model = ConvNet()
#   # NOTE: this is required for the ``fork`` method to work
#    model.share_memory()
#   processes = []
#    for rank in range(num_processes):
#        p = mp.Process(target=train, args=(model,))
#        p.start()
#        processes.append(p)
#    for p in processes:
#       p.join()
print('Script called by: ' , repr(__name__))
if __name__ == "__main__":
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    #call(["nvcc", "--version"]) #does not work
    #! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    if use_cuda :
        print('__Name of CUDA Devices:', torch.cuda.get_device_name(device))
    print('__Devices')
    #call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print("#######################################")
    
    
    print("--------Calling trainModel()")
    model = ConvNet()
    #model.share_memory()
    trainModel(model_input = model)
    
    #trainModel()
    #runMultipleThreads()