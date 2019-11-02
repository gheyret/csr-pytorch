# from train_model import print_cuda_information, train_model, evaluate_on_testing_set, create_dataloaders
from analytics.logger import TensorboardLogger, VisdomLogger
from early_stopping import EarlyStopping
from data.pytorch_dataloader_wav import Dataset, AudioDataLoader
from data.import_data import import_data_generated
import torch
from cnn_model import ConvNet2 as Net
from instructions_processor import InstructionsProcessor
import argparse

parser = argparse.ArgumentParser(description="CSR Pytorch")
parser.add_argument('--gsc_path', default='../data/GoogleSpeechCommands/wav_format/')
parser.add_argument('--generated_path', default='../data/GoogleSpeechCommands/generated/')
parser.add_argument('--continue_training', default=False)
parser.add_argument('--continue_training_model_path', default='./trained_models/checkpoint.pt')
parser.add_argument('--early_stopping_checkpoint_path', default='./trained_models/checkpoint.pt')
parser.add_argument('--end_early_for_profiling', default=False)
parser.add_argument('--end_early_batches', default=21)
parser.add_argument('--run_on_cpu', default=False)
parser.add_argument('--max_training_epochs', default=500)
parser.add_argument('--print_frequency', default=20)
parser.add_argument('--validation_patience', default=1)
parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--number_of_workers', default=8)
parser.add_argument('--batch_size', default=1200)
parser.add_argument('--early_stopping_delta', default=0.001)

if __name__ == "__main__":
    args = parser.parse_args()

    # Parameters
    model = Net()
    if args.continue_training:
        model.load_state_dict(torch.load(args.continue_training_model_path))

    # initialize the early_stopping object
    early_stopper = EarlyStopping(end_early=args.end_early_for_profiling, max_num_batches=args.end_early_batches,
                                  delta=args.early_stopping_delta,
                                  verbose=True, patience=args.validation_patience,
                                  checkpoint_path=args.early_stopping_checkpoint_path)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available() & (not args.run_on_cpu)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if not use_cuda:
        numberOfWorkers = 0
        pin_memory = False
    else:
        numberOfWorkers = args.number_of_workers
        pin_memory = True

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': numberOfWorkers,
              'pin_memory': pin_memory}

    # DATALOADERS Generated:
    test_path = args.generated_path + "test/"
    list_id_test, label_dict_test = import_data_generated(test_path)
    testing_set = Dataset(list_ids=list_id_test, wavfolder_path=test_path, label_dict=label_dict_test)
    testing_dataloader_gen = AudioDataLoader(testing_set, **params)

    validation_path = args.generated_path + "validation/"
    list_id_validation, label_dict_validation = import_data_generated(validation_path)
    validation_set = Dataset(list_ids=list_id_validation, wavfolder_path=validation_path,
                             label_dict=label_dict_validation)
    validation_dataloader_gen = AudioDataLoader(validation_set, **params)

    train_path = args.generated_path + "train/"
    list_id_train, label_dict_train = import_data_generated(train_path)
    training_set = Dataset(list_ids=list_id_train, wavfolder_path=train_path, label_dict=label_dict_train)
    training_dataloader_gen = AudioDataLoader(training_set, **params)
    from data.import_data import csv_to_list, csv_to_dict

    # DATALOADERS GSC:
    list_id_test = csv_to_list(args.gsc_path + "list_id_test.csv")
    label_dict_test = csv_to_dict(args.gsc_path + "dict_labels_test.csv")
    testing_set = Dataset(list_ids=list_id_test, wavfolder_path=args.gsc_path, label_dict=label_dict_test)
    testing_dataloader_gsc = AudioDataLoader(testing_set, **params)

    list_id_validation = csv_to_list(args.gsc_path + "list_id_validation.csv")
    label_dict_validation = csv_to_dict(args.gsc_path + "dict_labels_validation.csv")
    validation_set = Dataset(list_ids=list_id_validation, wavfolder_path=args.gsc_path,
                             label_dict=label_dict_validation)
    validation_dataloader_gsc = AudioDataLoader(validation_set, **params)

    list_id_train = csv_to_list(args.gsc_path + "list_id_train.csv")
    label_dict_train = csv_to_dict(args.gsc_path + "dict_labels_train.csv")
    training_set = Dataset(list_ids=list_id_train, wavfolder_path=args.gsc_path, label_dict=label_dict_train)
    training_dataloader_gsc = AudioDataLoader(training_set, **params)

    tensorboard_logger = TensorboardLogger()

    # ################################## GEN TRAIN ######################
    visdom_logger_train_gen = VisdomLogger("Training_gen", ["loss_train", "PER_train", "loss_val", "PER_val"], 10)

    processor_gen = InstructionsProcessor(model, training_dataloader_gen, validation_dataloader_gen,
                                          args.max_training_epochs,
                                          args.batch_size, args.learning_rate, use_cuda, early_stopper,
                                          tensorboard_logger,
                                          print_frequency=args.print_frequency)
    print("--------Calling train_model()")
    processor_gen.print_cuda_information(use_cuda, device)
    processor_gen.train_model(visdom_logger_train_gen)
    processor_gen.save_model("./trained_models/gen16000.pt")
    # processor_gen.load_model("./trained_models/gen16000.pt")

    print("Evaluating on generated data:")
    processor_gen.evaluate_model(testing_dataloader_gen, use_early_stopping=False, epoch=-1)
    print("Evaluating on recorded data:")
    processor_gen.evaluate_model(testing_dataloader_gsc, use_early_stopping=False, epoch=-1)

    # ################################# GSC TRAIN #########################

    early_stopper.reset()
    visdom_logger_train_gsc = VisdomLogger("Training_gsc", ["loss_train", "PER_train", "loss_val", "PER_val"], 10)
    processor_gsc = InstructionsProcessor(model, training_dataloader_gsc, validation_dataloader_gsc,
                                          args.max_training_epochs,
                                          args.batch_size, args.learning_rate, use_cuda, early_stopper,
                                          tensorboard_logger,
                                          print_frequency=args.print_frequency)
    print("--------Calling train_model()")
    processor_gsc.train_model(visdom_logger_train_gsc)
    processor_gsc.save_model("./trained_models/gsc16000.pt")
    # if not args.end_early_for_profiling:
    print("Evaluating on generated data:")
    processor_gsc.evaluate_model(testing_dataloader_gen, use_early_stopping=False, epoch=-1)
    print("Evaluating on recorded data:")
    processor_gsc.evaluate_model(testing_dataloader_gsc, use_early_stopping=False, epoch=-1)
    ##################################

    # model_to_evaluate.load_state_dict(torch.load(model_path_to_evaluate))
    # evaluate_on_testing_set(model_to_evaluate, testing_dataloader, criterion_ctc, beam_decoder)

    if use_cuda:
        print('Maximum GPU memory occupied by tensors:', torch.cuda.max_memory_allocated(device=None) / 1e9, 'GB')
        print('Maximum GPU memory managed by the caching allocator: ',
              torch.cuda.max_memory_cached(device=None) / 1e9, 'GB')