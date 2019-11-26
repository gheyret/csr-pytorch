import torch
import torch.nn as nn
import sys
import time
import numpy
from ctc_decoder import BeamSearchDecoder


class InstructionsProcessor(object):

    def __init__(self, model_input, training_dataloader, validation_dataloader, max_epochs_training, batch_size,
                 learning_rate,
                 using_cuda, early_stopping, tensorboard_logger, print_frequency=20):
        self.model = model_input

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.max_epochs_training = max_epochs_training
        self.batch_size = batch_size
        self.use_cuda = using_cuda
        self.early_stopping = early_stopping
        self.tensorboard_logger = tensorboard_logger
        self.decoder = BeamSearchDecoder()
        self.print_frequency = print_frequency

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CTCLoss(zero_infinity=True, reduction='mean')

        if self.use_cuda:
            self.model.cuda()

        self.batch_time = time.time()

    def process_batch_evaluation(self, local_data, losses, edit_distances):
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

        decoded_sequence, _, _, out_seq_len = self.decoder.beam_search_batch(outputs, output_lengths)
        edit_distance = self.decoder.compute_per(decoded_sequence, out_seq_len, local_targets, local_target_lengths,
                                                 len(local_target_lengths))
        edit_distances.append(edit_distance)
        return losses, edit_distances

    def process_batch_training(self, local_data, epoch, batch_i, n_training_batches, train_losses, edit_distances,
                               verbose=False, visdom_logger_train=None):
        local_batch, local_targets, local_input_percentages, local_target_lengths = local_data
        input_lengths = local_input_percentages.mul_(int(local_batch.size(3))).int()

        if epoch == 0 & batch_i == 0:
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

        if (batch_i + 1) % self.print_frequency == 0:
            decoded_sequence, _, _, out_seq_len = self.decoder.beam_search_batch(outputs, output_lengths)
            edit_distance = self.decoder.compute_per(decoded_sequence, out_seq_len, local_targets, local_target_lengths,
                                                     len(local_target_lengths))
            evaluated_label = decoded_sequence[0][0][0:out_seq_len[0][0]]
            true_label = local_targets[0, :local_target_lengths[0]]

            if visdom_logger_train is not None:
                visdom_logger_train.add_value(["loss_train", "PER_train"], [batch_loss, edit_distance])
                visdom_logger_train.update()
            train_losses.append(batch_loss)
            edit_distances.append(edit_distance)


            self.tensorboard_logger.update_scalar('continuous/loss', batch_loss)
            # visdom_logger.update(batch_loss, edit_distance)
            if verbose:
                print('Evaluated: ', evaluated_label)
                print('True:      ', true_label)
                self.print_metrics(current_epoch=epoch, current_batch=batch_i, total_batches=n_training_batches,
                                   loss=loss,
                                   edit_distance=edit_distance, start_time=self.batch_time,
                                   max_epochs=self.max_epochs_training,
                                   batch_size=self.batch_size, print_frequency=self.print_frequency)
            self.batch_time = time.time()
        return train_losses, edit_distances

    def train_model(self, visdom_logger_train, verbose=False):
        print('Function train_model called by: ', repr(__name__))
        total_time = time.time()

        n_training_batches = len(self.training_dataloader)
        print("Starting training:")
        self.batch_time = time.time()
        for epoch in range(self.max_epochs_training):
            # Training
            self.model.train()
            train_losses = []
            train_edit_distances = []
            if verbose:
                print("Epoch ", epoch, "/", self.max_epochs_training, " starting.")
            for batch_i, (local_data) in enumerate(self.training_dataloader, 0):
                train_losses, train_edit_distances = self.process_batch_training(local_data, epoch, batch_i,
                                                                                 n_training_batches, train_losses,
                                                                                 train_edit_distances, verbose, visdom_logger_train)

                self.early_stopping.exit_program_early()
                if self.early_stopping.stop_program:
                    break
            train_loss = numpy.average(train_losses)
            train_edit_distance = numpy.average(train_edit_distances)
            # visdom_logger_train.add_value(["loss_train", "PER_train"], [train_loss, train_edit_distance])
            if self.early_stopping.stop_program:
                break

            # Validation
            self.evaluate_model(self.validation_dataloader, use_early_stopping=True, epoch=epoch,
                                visdom_logger=None, verbose=verbose)
            #visdom_logger_train.update()
            if self.early_stopping.stop_training_early:
                print("Early stopping")
                break
        print('Finished Training')
        tot_time = time.time() - total_time
        print("Total training time {:.0f}h, {:.0f}m, {:.0f}s".format(numpy.floor(tot_time/3600),
                                                                     numpy.floor((tot_time % 3600)/60),
                                                                     numpy.floor(((tot_time % 3600) % 60))))

    def evaluate_model(self, data_dataloader, use_early_stopping, epoch, visdom_logger=None, mode="", verbose=False):
        n_dataloader_batches = len(data_dataloader)
        with torch.set_grad_enabled(False):
            self.model.eval()
            self.batch_time = time.time()
            eval_losses = []
            eval_edit_distances = []
            for (local_data) in data_dataloader:
                eval_losses, eval_edit_distances = self.process_batch_evaluation(local_data, eval_losses,
                                                                                 eval_edit_distances)
            eval_loss = numpy.average(eval_losses)
            eval_edit_distance = numpy.average(eval_edit_distances)
            if visdom_logger is not None:
                visdom_logger.add_value(["loss_val", "PER_val"], [eval_loss, eval_edit_distance])
            if verbose:
                print('---------------------------------------------------------------------------------------------')
            if use_early_stopping:
                self.early_stopping(eval_loss, self.model)
            if verbose:
                self.print_metrics(current_epoch=epoch, current_batch=n_dataloader_batches - 1,
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

    def print_metrics(self, current_epoch, current_batch, total_batches, loss, edit_distance, start_time, max_epochs,
                      batch_size, print_frequency):
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Edit Distance: {:.4f}, Time: {:.2f}s, Sample/s: {:.2f}'
              .format(current_epoch + 1, max_epochs, current_batch + 1, total_batches, loss.item(),
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
