# from train_model import print_cuda_information, train_model, evaluate_on_testing_set, create_dataloaders
from analytics.logger import TensorboardLogger, VisdomLogger
from early_stopping import EarlyStopping
from data.pytorch_dataloader_wav import Dataset, AudioDataLoader
from data.import_data import import_data_generated, import_data_libri_speech, import_data_gsc, concat_datasets, \
    randomly_partition_data, csv_to_list, csv_to_dict
import torch
from cnn_model import ConvNet2 as Net
from instructions_processor import InstructionsProcessor
import argparse

parser = argparse.ArgumentParser(description="CSR Pytorch")
parser.add_argument('--libri_path', default='../data/LibriSpeech/')
parser.add_argument('--vocab_path', default='../data/BritishEnglish_Reduced.xml')
parser.add_argument('--vocab_addition_path', default='../data/LibriSpeech/missing_words.dict')
parser.add_argument('--gsc_path', default='../data/GoogleSpeechCommands/wav_format/')
parser.add_argument('--generated_path', default='../data/GoogleSpeechCommands/generated/')
parser.add_argument('--continue_training', default=False)
parser.add_argument('--continue_training_model_path', default='./trained_models/checkpoint.pt')
parser.add_argument('--early_stopping_checkpoint_path', default='./trained_models/checkpoint.pt')
parser.add_argument('--end_early_for_profiling', default=False)
parser.add_argument('--end_early_batches', default=101)
parser.add_argument('--run_on_cpu', default=False)
parser.add_argument('--max_training_epochs', default=500)
parser.add_argument('--print_frequency', default=20)
parser.add_argument('--validation_patience', default=3)
parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--number_of_workers', default=8)
parser.add_argument('--batch_size', default=50)
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

    tensorboard_logger = TensorboardLogger()

    # Dataloaders:

    # DATALOADERS GSC:
    # Train
    list_id_train = csv_to_list(args.gsc_path + "list_id_train.csv")
    label_dict_train = csv_to_dict(args.gsc_path + "dict_labels_train.csv")
    training_set = Dataset(list_ids=list_id_train, wavfolder_path=args.gsc_path, label_dict=label_dict_train)
    training_dataloader_gsc = AudioDataLoader(training_set, **params)

    # Validation
    list_id_validation = csv_to_list(args.gsc_path + "list_id_validation.csv")
    label_dict_validation = csv_to_dict(args.gsc_path + "dict_labels_validation.csv")
    validation_set = Dataset(list_ids=list_id_validation, wavfolder_path=args.gsc_path,
                             label_dict=label_dict_validation)
    validation_dataloader_gsc = AudioDataLoader(validation_set, **params)

    # Test
    list_id_test = csv_to_list(args.gsc_path + "list_id_test.csv")
    label_dict_test = csv_to_dict(args.gsc_path + "dict_labels_test.csv")
    testing_set = Dataset(list_ids=list_id_test, wavfolder_path=args.gsc_path, label_dict=label_dict_test)
    testing_dataloader_gsc = AudioDataLoader(testing_set, **params)


    # DATALOADERS LibriSpeech:
    # Train & Validation
    libri_speech_dev_path_1 = args.libri_path + "train-clean-100/"  # "dev-clean/"
    libri_speech_dev_path_2 = args.libri_path + "train-clean-360/"  # "dev-clean/"

    list_id_1, label_dict_1, missing_words_1 = import_data_libri_speech(dataset_path=libri_speech_dev_path_1, vocabulary_path=args.vocab_path, vocabulary_path_addition=args.vocab_addition_path)
    list_id_2, label_dict_2, missing_words_2 = import_data_libri_speech(dataset_path=libri_speech_dev_path_2, vocabulary_path=args.vocab_path, vocabulary_path_addition=args.vocab_addition_path)

    list_id_train, label_dict_train, wav_path_train = concat_datasets(list_id_1, list_id_2,
                                                    label_dict_1, label_dict_2,
                                                    libri_speech_dev_path_1, libri_speech_dev_path_2)
    training_set = Dataset(list_ids=list_id_train, wavfolder_path=wav_path_train,
                           label_dict=label_dict_train)
    training_dataloader_ls = AudioDataLoader(training_set, **params)


    libri_speech_path_validation = args.libri_path + "dev-clean/"  # "dev-clean/"
    list_id_validation, label_dict_validation, missing_words_validation = import_data_libri_speech(
        dataset_path=libri_speech_path_validation, vocabulary_path=args.vocab_path)
    validation_set = Dataset(list_ids=list_id_validation, wavfolder_path=libri_speech_path_validation,
                             label_dict=label_dict_validation)
    validation_dataloader_ls = AudioDataLoader(validation_set, **params)


    #   Testing:
    libri_speech_test_path = args.libri_path + "test-clean/"
    list_id_test, label_dict_test, _ = import_data_libri_speech(dataset_path=libri_speech_test_path,
                                                      vocabulary_path=args.vocab_path)
    testing_set = Dataset(list_ids=list_id_test, wavfolder_path=libri_speech_test_path, label_dict=label_dict_test)
    testing_dataloader_ls = AudioDataLoader(testing_set, **params)



    # Processor:
    # ["loss_train", "PER_train", "loss_val", "PER_val"]
    visdom_logger_train_ls = VisdomLogger("Training LS dev", ["loss_train", "PER_train"], 10)

    processor_ls = InstructionsProcessor(model, training_dataloader_ls, validation_dataloader_ls,
                                          args.max_training_epochs,
                                          args.batch_size, args.learning_rate, use_cuda, early_stopper,
                                          tensorboard_logger,
                                          print_frequency=args.print_frequency)

    print("--------Calling train_model()")
    processor_ls.load_model("./trained_models/LS_base.pt")
    #processor_ls.save_model("./trained_models/LS_base.pt")
    processor_ls.train_model(visdom_logger_train_ls, verbose=True)
    processor_ls.save_model("./trained_models/LS_base.pt")

    print("Evaluating on test data:")
    processor_ls.evaluate_model(testing_dataloader_ls, use_early_stopping=False, epoch=-1)

    print(len(missing_words_1))
    
    print(len(list_id_1))
    print(len(missing_words_2))
    print(len(list_id_2))


    if use_cuda:
        print('Maximum GPU memory occupied by tensors:', torch.cuda.max_memory_allocated(device=None) / 1e9, 'GB')
        print('Maximum GPU memory managed by the caching allocator: ',
              torch.cuda.max_memory_cached(device=None) / 1e9, 'GB')
