# from train_model import print_cuda_information, train_model, evaluate_on_testing_set, create_dataloaders
from analytics.logger import TensorboardLogger, VisdomLogger
from early_stopping import EarlyStopping
from data.pytorch_dataloader_wav import Dataset, AudioDataLoader
from data.import_data import import_data_libri_speech, randomly_partition_data, csv_to_list, csv_to_dict, \
    print_label_distribution, concat_datasets
import torch
from models.ConvNet8 import ConvNet8 as Net
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
parser.add_argument('--batch_size', default=15)
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

    # DATALOADERS LibriSpeech:
    # Train & Validation
    # Train on dev-clean val on part of dev-clean
    libri_speech_dev_path = args.libri_path + "dev-clean/"
    list_id, label_dict, missing_words = import_data_libri_speech(dataset_path=libri_speech_dev_path,
                                                                  vocabulary_path=args.vocab_path,
                                                                  vocabulary_path_addition=args.vocab_addition_path)
    list_id_train, list_id_validation, label_dict_train, label_dict_validation = randomly_partition_data(0.9, list_id,
                                                                                                         label_dict)
    training_set = Dataset(list_ids=list_id_train, wavfolder_path=libri_speech_dev_path,
                           label_dict=label_dict_train)
    training_dataloader_ls = AudioDataLoader(training_set, **params)
    validation_set = Dataset(list_ids=list_id_validation, wavfolder_path=libri_speech_dev_path,
                             label_dict=label_dict_validation)
    validation_dataloader_ls = AudioDataLoader(validation_set, **params)

    #   Testing:
    libri_speech_test_path = args.libri_path + "test-clean/"
    list_id_test, label_dict_test, _ = import_data_libri_speech(dataset_path=libri_speech_test_path,
                                                      vocabulary_path=args.vocab_path)
    testing_set = Dataset(list_ids=list_id_test, wavfolder_path=libri_speech_test_path, label_dict=label_dict_test)
    testing_dataloader_ls = AudioDataLoader(testing_set, **params)

    print(len(list_id_train))
    print(len(list_id_validation))

    # Processor:
    # ["loss_train", "PER_train", "loss_val", "PER_val"]

    from models.ConvNet2 import ConvNet2
    from models.ConvNet3 import ConvNet3
    from models.ConvNet4 import ConvNet4
    from models.ConvNet5 import ConvNet5
    from models.ConvNet6 import ConvNet6
    from models.ConvNet7 import ConvNet7
    from models.ConvNet8 import ConvNet8
    from models.ConvNet9 import ConvNet9
    from models.ConvNet10 import ConvNet10
    from models.ConvNet11 import ConvNet11
    from models.ConvNet12 import ConvNet12
    from models.ConvNet13 import ConvNet13
    from models.ConvNet14 import ConvNet14
    from models.ConvNet15 import ConvNet15

    model_num = [15]
    for i_model, model in enumerate([ConvNet15()]):

        model_name = "ConvNet" + str(model_num[i_model])
        visdom_logger_train_ls = VisdomLogger("LS dev " + model_name, ["loss_train", "PER_train", "loss_val", "PER_val"], 10)
        processor_ls = InstructionsProcessor(model, training_dataloader_ls, validation_dataloader_ls,
                                              args.max_training_epochs,
                                              args.batch_size, args.learning_rate, use_cuda, early_stopper,
                                              tensorboard_logger,
                                              print_frequency=args.print_frequency)
        print("--------Calling train_model()")
        #processor_ls.load_model("./trained_models/LS_ConvNet" + str(model_num[i_model]) + ".pt")
        processor_ls.train_model(visdom_logger_train_ls, verbose=True)
        processor_ls.save_model("./trained_models/LS_" + model_name + ".pt")
        print("Evaluating on test data for " + model_name + ":")
        processor_ls.evaluate_model(testing_dataloader_ls, use_early_stopping=False, epoch=-1, verbose=True)
        early_stopper.reset()

    if use_cuda:
        print('Maximum GPU memory occupied by tensors:', torch.cuda.max_memory_allocated(device=None) / 1e9, 'GB')
        print('Maximum GPU memory managed by the caching allocator: ',
              torch.cuda.max_memory_cached(device=None) / 1e9, 'GB')
