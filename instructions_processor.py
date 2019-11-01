import torch
import torch.nn as nn
import sys
import time
import numpy
from ctc_decoder import BeamSearchDecoder





class InstructionsProcessor(object):

    def __init__(self, model_input, training_generator, validation_generator, max_epochs_training, batch_size, learning_rate,
                using_cuda, early_stopping, tensorboard_logger, visdom_logger, print_frequency=20):
        self.model = model_input

        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.max_epochs_training = max_epochs_training
        self.batch_size = batch_size
        self.use_cuda = using_cuda
        self.early_stopping = early_stopping
        self.tensorboard_logger = tensorboard_logger
        self.visdom_logger = visdom_logger
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

    def process_batch_training(self, local_data, epoch, batch_i, n_training_batches):
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
            # edit_distance, _ = compute_edit_distance(outputs, local_targets, local_target_lengths, 0)
            print('Evaluated: ', evaluated_label)
            print('True:      ', true_label)
            self.tensorboard_logger.update_scalar('continuous/loss', batch_loss)
            self.visdom_logger.update(batch_loss, edit_distance)

            self.print_metrics(current_epoch=epoch, current_batch=batch_i, total_batches=n_training_batches, loss=loss,
                          edit_distance=edit_distance, start_time=self.batch_time, max_epochs=self.max_epochs_training,
                          batch_size=self.batch_size, print_frequency=self.print_frequency)
            self.batch_time = time.time()

    def train_model(self):
        print('Function train_model called by: ', repr(__name__))
        total_time = time.time()
        train_losses = []

        n_training_batches = len(self.training_generator)
        n_validation_batches = len(self.validation_generator)
        print("Starting training:")
        self.batch_time = time.time()
        for epoch in range(self.max_epochs_training):
            # Training
            self.model.train()
            print("Epoch ", epoch, "/", self.max_epochs_training, " starting.")
            for batch_i, (local_data) in enumerate(self.training_generator, 0):
                self.process_batch_training(local_data, epoch, batch_i, n_training_batches)
                self.early_stopping.exit_program_early()
                if self.early_stopping.stop_program:
                    break
            if self.early_stopping.stop_program:
                break

            # Validation
            self.evaluate_model(self.validation_generator, use_early_stopping=True, epoch=epoch)

            if self.early_stopping.stop_training_early:
                print("Early stopping")
                break
        print('Finished Training')
        print("Total training time: ", (time.time() - total_time), " s")

    def evaluate_model(self, data_generator, use_early_stopping, epoch, mode=""):
        n_generator_batches = len(data_generator)
        with torch.set_grad_enabled(False):
            self.model.eval()
            self.batch_time = time.time()
            eval_losses = []
            eval_edit_distances = []
            for (local_data) in data_generator:
                eval_losses, eval_edit_distances = self.process_batch_evaluation(local_data, eval_losses, eval_edit_distances)
            eval_loss = numpy.average(eval_losses)
            eval_edit_distance = numpy.average(eval_edit_distances)

            print('---------------------------------------------------------------------------------------------')
            if use_early_stopping:
                self.early_stopping(eval_loss, self.model)
            self.print_metrics(current_epoch=epoch, current_batch=n_generator_batches - 1,
                               total_batches=n_generator_batches,
                               loss=eval_loss, edit_distance=eval_edit_distance, start_time=self.batch_time,
                               max_epochs=self.max_epochs_training,
                               batch_size=self.batch_size, print_frequency=n_generator_batches)
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
