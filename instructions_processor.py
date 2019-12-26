import torch
import torch.nn as nn
import sys
import time
import numpy

from analytics.logger import VisdomLogger
from ctc_decoder import BeamSearchDecoder
import math


def cyclical_lr(stepsize, max_lr=1e-3, min_lr=3e-4):
    '''
    Cyclical learning rate. Starts at min_lr and increases linearly to max_lr over stepsize iteartions. Then it decreases
    linearily to min_lr over stepsize iterations and repeats. The max_lr is for each cycle decreased by scaler.
    '''
    # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 0.85 ** (x - 1)

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def decaying_lr(stepsize=250000, max_lr=1e-3, min_lr=3e-4):
    '''
    Linearly decaying learning_rate. Starts at max_lr and decreases linearily to min_lr over stepsize iterations.
    '''
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * max(0, (1 - it / stepsize))
    return lr_lambda


class InstructionsProcessor(object):

    def __init__(self, model_input, training_dataloader, validation_dataloader, training_dataloader_ordered=None,
                 batch_size=32, max_epochs_training=500, max_epochs_training_ordered=5, learning_rate=1e-3,
                 learning_rate_mode='fixed',
                 min_learning_rate_factor=6.0,
                 learning_rate_step_size=5,
                 num_classes=None,
                 using_cuda=False, early_stopping=None,
                 tensorboard_logger=None, mini_epoch_length=20, visdom_logger_train=None, track_learning_rate=False):
        self.model = model_input
        self.batch_size = batch_size
        self.use_cuda = using_cuda
        self.early_stopping = early_stopping
        self.tensorboard_logger = tensorboard_logger
        self.decoder = BeamSearchDecoder(num_classes)
        self.mini_epoch_length = mini_epoch_length
        self.max_epochs_training = max_epochs_training
        self.training_dataloader = training_dataloader
        if training_dataloader_ordered is not None:
            self.training_dataloader_ordered = training_dataloader_ordered
            self.max_epochs_training_ordered = max_epochs_training_ordered
        self.validation_dataloader = validation_dataloader
        self.total_time = 0
        self.epoch = -1
        if visdom_logger_train is not None:
            self.visdom_logger_train = visdom_logger_train
            logger_title = self.visdom_logger_train.title
        else:
            logger_title = "Learning rate of training"

        self.track_learning_rate = track_learning_rate
        if track_learning_rate:
            self.visdom_learning_rate_logger = VisdomLogger(logger_title, ["learning_rate", "learning_rate"], 10)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')
        if learning_rate_mode is 'cyclic':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1)  # , weight_decay=1e-4
            step_size = learning_rate_step_size * len(self.training_dataloader)
            clr = cyclical_lr(step_size, max_lr=learning_rate, min_lr=learning_rate / min_learning_rate_factor)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])
        elif learning_rate_mode is 'decaying':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1)  # , weight_decay=1e-4
            step_size = learning_rate_step_size * len(self.training_dataloader)
            dec = decaying_lr(stepsize=step_size, max_lr=learning_rate, min_lr=learning_rate / min_learning_rate_factor)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [dec])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # , weight_decay=1e-4
            self.scheduler = None

        if self.use_cuda:
            self.model.cuda()

        self.batch_time = time.time()

    def test_learning_rate_lambda(self, learning_rate_mode='cyclic', max_lr=1e-3, factor=6.0, step_size_factor=1,
                                  test_epochs=10):
        '''
        Used to get an overview  of how the learning_rate will behave over epochs
        '''
        if learning_rate_mode is 'cyclic':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1)  # , weight_decay=1e-4
            step_size = step_size_factor * len(self.training_dataloader)
            clr = cyclical_lr(step_size, max_lr=max_lr, min_lr=max_lr / factor)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [clr])

        if learning_rate_mode is 'decaying':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1)  # , weight_decay=1e-4
            step_size = step_size_factor * len(self.training_dataloader)
            dec = decaying_lr(stepsize=step_size, max_lr=max_lr, min_lr=max_lr / factor)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, [dec])

        learning_rates = []
        for i in range(test_epochs * len(self.training_dataloader)):
            for g in self.optimizer.param_groups:
                learning_rate = g['lr']
            learning_rates.append(learning_rate)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(learning_rates)
        plt.show()

    def find_learning_rate(self, start_lr, end_lr, lr_find_epochs):
        '''
        Used to find the a range of acceptable learning rates.
        :param start_lr: Starting learning rate. e.g. 1e-7
        :param end_lr: Ending learning rate. e.g. 1e-1
        :param lr_find_epochs: Over how many epochs the test should be performed. Over these epochs the learning rate
        will increase form start_lr to end_lr in an exponential fashion.
        :return:
        '''

        lr_lambda = lambda x: math.exp(
            x * math.log(end_lr / start_lr) / (lr_find_epochs * (len(self.training_dataloader))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        for g in self.optimizer.param_groups:
            old_lr = g['lr']
            g['lr'] = start_lr
        print(start_lr)

        lr_find_loss = []
        lr_find_lr = []

        train_losses = []
        train_edit_distances = []

        iteration = 0
        smoothing = 0.05
        n_training_batches = len(self.training_dataloader)
        for i in range(lr_find_epochs):
            print("epoch {}".format(i))
            for batch_i, (local_data) in enumerate(self.training_dataloader, 0):
                outputs, output_lengths, local_targets, local_target_lengths, batch_loss = self.process_batch_training(
                    local_data, batch_i)
                train_losses, edit_distances = self.evaluate_training_progress(batch_i, outputs,
                                                                               output_lengths, local_targets,
                                                                               local_target_lengths,
                                                                               batch_loss, train_losses,
                                                                               train_edit_distances,
                                                                               n_training_batches, verbose=False)

                loss = train_losses[-1]

                # Update LR
                scheduler.step()

                lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
                print(lr_step)
                lr_find_lr.append(lr_step)

                # smooth the loss
                if iteration == 0:
                    lr_find_loss.append(loss)
                else:
                    loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                    lr_find_loss.append(loss)

                iteration += 1
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.semilogx(lr_find_lr, lr_find_loss)
        plt.subplot(2, 1, 2)
        plt.plot(lr_find_lr)
        plt.show()

        for g in self.optimizer.param_groups:
            g['lr'] = old_lr

    def process_batch_evaluation(self, local_data, losses, edit_distances):
        '''
        Perform the forward pass for the batch for the purpose of evaluating performance.
        :param local_data:
        :param losses:
        :param edit_distances:
        :return:
        '''
        local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
        input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()
        # Transfer to GPU
        if self.use_cuda:  # On GPU
            local_batch, local_targets = local_batch.cuda(non_blocking=True), local_targets.cuda(
                non_blocking=True)
        # Run the forward pass
        outputs, output_lengths = self.model(local_batch, input_lengths)
        loss = self.criterion(outputs, local_targets, output_lengths, local_target_lengths)  # CTC loss function

        losses.append(loss.item())

        decoded_sequence, scores, _, out_seq_len = self.decoder.beam_search_batch(outputs, output_lengths)
        edit_distance = self.decoder.compute_per(decoded_sequence, out_seq_len, local_targets, local_target_lengths,
                                                 len(local_target_lengths))

        edit_distances.append(edit_distance)
        return losses, edit_distances

    def process_batch_training(self, local_data, batch_i):
        '''
        Perform the forward and backward passes for the batch for the purpose of training.
        :param local_data: Input data for this batch.
        :param batch_i: Current batch index.
        :return:
        '''
        local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
        input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()

        if self.epoch == 0 & batch_i == 0:
            logger_args = [local_batch, input_lengths]
            # logger.add_model_graph(model, logger_args)  # Can't get model graph to work with RNN...

        # Transfer to GPU
        if self.use_cuda:  # On GPU
            local_batch, local_targets = local_batch.cuda(non_blocking=True), local_targets.cuda(non_blocking=True)
        # Run the forward pass
        outputs, output_lengths = self.model(local_batch, input_lengths)
        # Compute loss
        loss = self.criterion(outputs, local_targets, output_lengths, local_target_lengths)

        batch_loss = loss.item()
        # Backpropagation and perform Adam optimisation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return outputs, output_lengths, local_targets, local_target_lengths, batch_loss

    def evaluate_training_progress(self, batch_i, outputs, output_lengths, local_targets,
                                   local_target_lengths, batch_loss, train_losses, edit_distances, n_training_batches,
                                   verbose=False, visdom_logger_train=None):
        '''
        Function called during training that evaluates the current progress of the model and collects data for tracking.
        :param batch_i: Current batch index
        :param outputs: The output tensor from the previous batch evaluation. Expects [T x N x C]
        :param output_lengths: The lengths of the sequences.
        :param local_targets: The true labels of the previous batch.
        :param local_target_lengths: The length of the samples in local_targets
        :param visdom_logger_train: The logger used to collect data.
        :param batch_loss: The computed loss for this batch.
        :param train_losses: The list containing the previous losses for this epoch.
        :param edit_distances: The list containing the previous edit distances for this epoch
        :param verbose:
        :param epoch: Current epoch index
        :param n_training_batches: How many batches that is expected to be evaluated in this epoch.
        :return:
        '''
        decoded_sequence, _, _, out_seq_len = self.decoder.beam_search_batch(outputs, output_lengths)
        edit_distance = self.decoder.compute_per(decoded_sequence, out_seq_len, local_targets, local_target_lengths,
                                                 len(local_target_lengths))
        evaluated_label = decoded_sequence[0][0][0:out_seq_len[0][0]]
        true_label = local_targets[0, :local_target_lengths[0]]

        if visdom_logger_train is not None:
            visdom_logger_train.add_value(["loss_train", "PER_train"], [batch_loss, edit_distance])
        train_losses.append(batch_loss)
        edit_distances.append(edit_distance)

        self.tensorboard_logger.update_scalar('continuous/loss', batch_loss)
        # visdom_logger.update(batch_loss, edit_distance)
        if verbose:
            print('Evaluated: ', evaluated_label)
            print('True:      ', true_label)
            self.print_metrics(current_batch=batch_i, total_batches=n_training_batches,
                               loss=batch_loss,
                               edit_distance=edit_distance, start_time=self.batch_time,
                               max_epochs=self.max_epochs_training,
                               batch_size=self.batch_size, print_frequency=self.mini_epoch_length)
            for g in self.optimizer.param_groups:
                print('Learning rate: ', g['lr'])
        self.batch_time = time.time()
        return train_losses, edit_distances

    def train_model(self, mini_epoch_validation_partition_size, mini_epoch_evaluate_validation,
                    mini_epoch_early_stopping, ordered=False, verbose=False):
        '''
        Function called to initiate training on self.model using the parameters specified in initialization.
        :param visdom_logger_train: The logger to be used to collect data.
        :param mini_epoch_validation_partition_size: How large proportion of the validation set should be evaluated?
        :param mini_epoch_evaluate_validation: True if the validation set should be evaluated after a certain number
        (self.mini_epoch_length) of batches. Useful for gaining log data more frequently over very long epochs.
        :param ordered: True if this training is supposed to be done on the ordered dataloader
        :param verbose:
        :return:
        '''
        # Todo: Clean up the logging to make this section easier to read.
        if ordered:
            max_epochs_training = self.max_epochs_training_ordered
            training_dataloader = self.training_dataloader_ordered
        else:
            max_epochs_training = self.max_epochs_training
            training_dataloader = self.training_dataloader

        if mini_epoch_evaluate_validation:
            visdom_logger_train_evaluate = self.visdom_logger_train
            visdom_logger_val_evaluate = None
        else:
            visdom_logger_train_evaluate = None
            visdom_logger_val_evaluate = self.visdom_logger_train

        print('Function train_model called by: ', repr(__name__))
        if self.total_time == 0:  # Set the total_time to the start of the first training.
            self.total_time = time.time()



        n_training_batches = len(training_dataloader)
        print("Starting training:")
        self.batch_time = time.time()
        for epoch in range(max_epochs_training):
            self.epoch += 1
            # Training
            self.model.train()
            train_losses = []
            train_edit_distances = []
            if verbose:
                print("Epoch ", self.epoch, "/", max_epochs_training, " starting.")
            for batch_i, (local_data) in enumerate(training_dataloader, 0):
                outputs, output_lengths, local_targets, local_target_lengths, batch_loss = self.process_batch_training(
                    local_data, batch_i)

                if (batch_i + 1) % self.mini_epoch_length == 0:
                    train_losses, train_edit_distances = self.evaluate_training_progress(batch_i, outputs,
                                                                                         output_lengths, local_targets,
                                                                                         local_target_lengths,
                                                                                         batch_loss, train_losses,
                                                                                         train_edit_distances,
                                                                                         n_training_batches, verbose,
                                                                                         visdom_logger_train=visdom_logger_train_evaluate)
                    if mini_epoch_evaluate_validation:
                        self.evaluate_model(self.validation_dataloader, use_early_stopping=mini_epoch_early_stopping,
                                            visdom_logger=self.visdom_logger_train, verbose=verbose,
                                            part=mini_epoch_validation_partition_size)
                        if self.track_learning_rate:
                            for g in self.optimizer.param_groups:
                                learning_rate = g['lr']
                            self.visdom_learning_rate_logger.add_value(["learning_rate"], [learning_rate])
                            self.visdom_learning_rate_logger.update()
                self.early_stopping.exit_program_early()
                if self.early_stopping.stop_program:
                    break

            if not mini_epoch_evaluate_validation:
                train_loss = numpy.average(train_losses)
                train_edit_distance = numpy.average(train_edit_distances)
                self.visdom_logger_train.add_value(["loss_train", "PER_train"], [train_loss, train_edit_distance])

            if self.early_stopping.stop_program:
                break

            # Validation
            self.evaluate_model(self.validation_dataloader, use_early_stopping=True,
                                visdom_logger=visdom_logger_val_evaluate, verbose=verbose)
            if (not mini_epoch_evaluate_validation) & self.track_learning_rate:
                for g in self.optimizer.param_groups:
                    learning_rate = g['lr']
                self.visdom_learning_rate_logger.add_value(["learning_rate"], [learning_rate])
                self.visdom_learning_rate_logger.update()

            if self.early_stopping.stop_training_early:
                print("Early stopping")
                break
        print('Finished Training')
        tot_time = time.time() - self.total_time
        print("Total training time {:.0f}h, {:.0f}m, {:.0f}s".format(numpy.floor(tot_time / 3600),
                                                                     numpy.floor((tot_time % 3600) / 60),
                                                                     numpy.floor(((tot_time % 3600) % 60))))



    def evaluate_model(self, data_dataloader, use_early_stopping, visdom_logger=None, verbose=False, part=1.0):
        n_dataloader_batches = len(data_dataloader)
        # Todo: Add comments
        with torch.set_grad_enabled(False):
            self.model.eval()
            self.batch_time = time.time()
            eval_losses = []
            eval_edit_distances = []
            num_batches = len(data_dataloader)
            for batch_i, (local_data) in enumerate(data_dataloader):
                eval_losses, eval_edit_distances = self.process_batch_evaluation(local_data, eval_losses,
                                                                                 eval_edit_distances)
                if batch_i / num_batches > part:
                    break
            self.model.train()
            eval_loss = numpy.average(eval_losses)
            eval_edit_distance = numpy.average(eval_edit_distances)
            if visdom_logger is not None:
                visdom_logger.add_value(["loss_val", "PER_val"], [eval_loss, eval_edit_distance])
                visdom_logger.update()
            if verbose:
                print('---------------------------------------------------------------------------------------------')
            if use_early_stopping:
                self.early_stopping(eval_loss, self.model)
            if verbose:
                self.print_metrics(current_batch=batch_i,
                                   total_batches=n_dataloader_batches,
                                   loss=eval_loss, edit_distance=eval_edit_distance, start_time=self.batch_time,
                                   max_epochs=self.max_epochs_training,
                                   batch_size=self.batch_size, print_frequency=n_dataloader_batches)
                print('#############################################################################################')

    def print_cuda_information(self, use_cuda, device):
        print("############ CUDA Information #############")
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        if use_cuda:
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

    def print_metrics(self, current_batch, total_batches, loss, edit_distance, start_time, max_epochs,
                      batch_size, print_frequency):
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Edit Distance: {:.4f}, Time: {:.2f}s, Sample/s: {:.2f}'
              .format(self.epoch, max_epochs, current_batch + 1, total_batches, loss,
                      edit_distance, (time.time() - start_time),
                      batch_size * print_frequency / (time.time() - start_time)))

    def set_model(self, model_input):
        self.model = model_input

    def get_model(self):
        return self.model

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)
        print("Saved model at path: ", model_save_path)

    def load_model(self, model_load_path):
        self.model.load_state_dict(torch.load(model_load_path))
        print("Loaded model at path: ", model_load_path)
