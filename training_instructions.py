from analytics.logger import TensorboardLogger, VisdomLogger
from early_stopping import EarlyStopping
from data.pytorch_dataloader_wav import Dataset, AudioDataLoader
from data.data_importer import import_data_libri_speech, randomly_partition_data, csv_to_list, csv_to_dict, \
    print_label_distribution, concat_datasets, order_data_by_length, get_num_classes, import_data_common_voice_v1, \
    import_data_common_voice_1087
import torch
from models.ConvNet8 import ConvNet8 as Net
from instructions_processor import InstructionsProcessor
import argparse
import time

parser = argparse.ArgumentParser(description="CSR Pytorch")
parser.add_argument('--libri_path', default='../data/LibriSpeech/')
parser.add_argument('--commonvoice_v1_path', default='../data/CommonVoice/cv_corpus_v1/')
parser.add_argument('--commonvoice_1087_path', default='../data/CommonVoice/en/')
parser.add_argument('--gsc_path', default='../data/GoogleSpeechCommands/wav_format/')
parser.add_argument('--generated_path', default='../data/GoogleSpeechCommands/generated/')

parser.add_argument('--vocab_path', default='../data/missing_words.dict')  # BritishEnglish_Reduced.xml
parser.add_argument('--vocab_is_xml', default=False)
parser.add_argument('--vocab_addition_path', default=None)  # '../data/LibriSpeech/missing_words.dict'

parser.add_argument('--continue_training', default=False)
parser.add_argument('--continue_training_model_path', default='./trained_models/checkpoint.pt')
parser.add_argument('--early_stopping_checkpoint_path', default='./trained_models/checkpoint.pt')

parser.add_argument('--end_early_for_profiling', default=False)
parser.add_argument('--end_early_batches', default=101)

parser.add_argument('--run_on_cpu', default=False)
parser.add_argument('--max_training_epochs', default=500)
parser.add_argument('--max_training_epochs_ordered', default=1)

parser.add_argument('--mini_epoch_length', default=150)
parser.add_argument('--mini_epoch_evaluate_validation', default=True)
parser.add_argument('--mini_epoch_early_stopping', default=False)
parser.add_argument('--mini_epoch_validation_partition_size', default=0.2)
parser.add_argument('--early_stopping_delta', default=0.001)
parser.add_argument('--validation_patience', default=2)

parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--learning_rate_mode', default='cyclic')  # static or cyclic or decaying
parser.add_argument('--min_learning_rate_factor', default=10.0)
parser.add_argument('--learning_rate_step_size', default=0.25)  # Epochs
parser.add_argument('--track_learning_rate', default=True)  # Track or not

parser.add_argument('--number_of_workers', default=8)
parser.add_argument('--batch_size', default=16)

parser.add_argument('--rnn_memory_type', default='GRU')  # GRU or LSTM or RNN
parser.add_argument('--rnn_bidirectional', default=True)  # GRU or LSTM bidirectional?
parser.add_argument('--non_linearity', default='ReLU')  # ReLU or Hardtanh

parser.add_argument('--input_type', default='features')  # features or raw or power
parser.add_argument('--num_features_input', default=120)  # features = Any, power = 257
parser.add_argument('--use_delta_features', default=False)  # Use of delta & delta delta features

parser.add_argument('--architecture_type', default='CTC')  # CTC or LAS
parser.add_argument('--label_type', default='letter')  # phoneme or letter
parser.add_argument('--training_libri_type', default='clean')  # dev, clean or other


if __name__ == "__main__":
    args = parser.parse_args()

    assert (not (args.input_type is 'raw') & (args.use_delta_features is True)), "Can't have raw input & delta features"
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

    dataloader_kwargs = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': numberOfWorkers,
              'pin_memory': pin_memory}
    dataloader_kwargs_ordered = dataloader_kwargs.copy()
    dataloader_kwargs_ordered["shuffle"] = False

    if args.use_delta_features:
        num_input_channels = 3
    else:
        num_input_channels = 1

    tensorboard_logger = TensorboardLogger()

    ids = ["letter_clean", "phoneme_clean"]
    label_types = ["letter", "phoneme"]

    for i, id in enumerate(ids):
        args.label_type = label_types[i]
        import_kwargs = {"vocabulary_path": args.vocab_path, "vocabulary_path_is_xml": args.vocab_is_xml,
                                                             "vocabulary_path_addition":args.vocab_addition_path,
                                                             "label_type": args.label_type}
        dataset_kwargs = {"num_features_input":args.num_features_input,
                                       "use_delta_features":args.use_delta_features, "input_type": args.input_type}
        model_kwargs = {"num_classes": get_num_classes(args.label_type), "num_features_input": args.num_features_input,
                        "num_input_channels": num_input_channels, "non_linearity": args.non_linearity,
                        "rnn_bidirectional": args.rnn_bidirectional,
                        "memory_type": args.rnn_memory_type, "input_type": args.input_type}
        processor_kwargs = {"batch_size": args.batch_size,
                            "max_epochs_training": args.max_training_epochs,
                             "max_epochs_training_ordered": args.max_training_epochs_ordered,
                             "learning_rate": args.learning_rate,
                             "learning_rate_mode": args.learning_rate_mode,
                             "min_learning_rate_factor": args.min_learning_rate_factor,
                             "learning_rate_step_size": args.learning_rate_step_size,
                             "num_classes": get_num_classes(args.label_type),
                             "using_cuda": use_cuda,
                            "early_stopping": early_stopper,
                             "tensorboard_logger": tensorboard_logger,
                             "mini_epoch_length": args.mini_epoch_length,
                             "track_learning_rate": args.track_learning_rate}


        # DATALOADERS LibriSpeech:
        # Train & Validation

        # Train on dev-clean val on part of dev-clean
        libri_speech_dev_clean_path = args.libri_path + "dev-clean/"
        libri_speech_dev_other_path = args.libri_path + "dev-other/"
        libri_speech_test_clean_path = args.libri_path + "test-clean/"
        libri_speech_test_other_path = args.libri_path + "test-other/"
        libri_speech_train_100_path = args.libri_path + "train-clean-100/"
        libri_speech_train_360_path = args.libri_path + "train-clean-360/"
        libri_speech_train_500_path = args.libri_path + "train-other-500/"

        if args.training_libri_type is "dev":
            collection_ls_dev_clean, missing_words_ls_dev_clean = import_data_libri_speech(dataset_path=libri_speech_dev_clean_path, **import_kwargs)
            collection_train, collection_validation = randomly_partition_data(0.9, collection_ls_dev_clean)
        elif args.training_libri_type is "clean":
            collection_ls_train_100, missing_words_ls_train_100 = import_data_libri_speech(dataset_path=libri_speech_train_100_path, **import_kwargs)
            collection_ls_train_360, missing_words_ls_train_360 = import_data_libri_speech(dataset_path=libri_speech_train_360_path, **import_kwargs)

            collection_ls_dev_clean, missing_words_ls_dev_clean = import_data_libri_speech(dataset_path=libri_speech_dev_clean_path, **import_kwargs)

            collection_train = concat_datasets(collection_ls_train_100, collection_ls_train_360)
            collection_validation = collection_ls_dev_clean
        elif args.training_libri_type is "other":
            collection_ls_train_100, missing_words_ls_train_100 = import_data_libri_speech(dataset_path=libri_speech_train_100_path, **import_kwargs)
            collection_ls_train_360, missing_words_ls_train_360 = import_data_libri_speech(dataset_path=libri_speech_train_360_path, **import_kwargs)
            collection_ls_train_500, missing_words_ls_train_500 = import_data_libri_speech(dataset_path=libri_speech_train_500_path, **import_kwargs)

            collection_ls_dev_clean, missing_words_ls_dev_clean = import_data_libri_speech(dataset_path=libri_speech_dev_clean_path, **import_kwargs)
            collection_ls_dev_other, missing_words_ls_dev_other = import_data_libri_speech(dataset_path=libri_speech_dev_other_path, **import_kwargs)

            collection_train = concat_datasets(collection_ls_train_100, collection_ls_train_360)
            collection_train = concat_datasets(collection_train, collection_ls_train_500)

            collection_validation = concat_datasets(collection_ls_dev_clean, collection_ls_dev_other)

        #collection_cv_train, collection_cv_val, collection_cv_test, missing_words_tot = import_data_common_voice_1087(
        #                                                        dataset_path=args.commonvoice_1087_path, **import_kwargs)
        #testing_set_cv = Dataset(collection=collection_cv_test, **dataset_kwargs)
        #testing_dataloader_cv = AudioDataLoader(testing_set_cv, **dataloader_kwargs)

        collection_train_ordered = order_data_by_length(collection_train)
        training_set_ordered = Dataset(collection=collection_train_ordered, **dataset_kwargs)
        training_dataloader_ordered = AudioDataLoader(training_set_ordered, **dataloader_kwargs_ordered)
        training_dataloader = AudioDataLoader(training_set_ordered, **dataloader_kwargs)

        validation_set = Dataset(collection=collection_validation, **dataset_kwargs)
        validation_dataloader = AudioDataLoader(validation_set, **dataloader_kwargs)

        #   Testing:
        collection_ls_test_clean, missing_words_ls_test_clean = import_data_libri_speech(dataset_path=libri_speech_test_clean_path, **import_kwargs)
        collection_ls_test_other, missing_words_ls_test_other = import_data_libri_speech(dataset_path=libri_speech_test_other_path, **import_kwargs)

        testing_set_ls_clean = Dataset(collection=collection_ls_test_clean, **dataset_kwargs)
        testing_dataloader_ls_clean = AudioDataLoader(testing_set_ls_clean, **dataloader_kwargs)

        testing_set_ls_other = Dataset(collection=collection_ls_test_other, **dataset_kwargs)
        testing_dataloader_ls_other = AudioDataLoader(testing_set_ls_other, **dataloader_kwargs)



        """
        missing_words_train = missing_words_1.append(missing_words_2)
    
        missing_words_validation = missing_words_train
    
        print("Training samples/missing: {}/{}. Val samples/missing {}/{}. Test samples/missing {}/{}.".format(
            len(list_id_train), len(missing_words_train), len(list_id_validation),
            len(missing_words_validation), len(list_id_test), len(missing_words_test)))
        """
        print("Training samples: {}. Val samples {}. Test samples {}.".format(
            training_dataloader.__len__()*args.batch_size, validation_dataloader.__len__()*args.batch_size, testing_dataloader_ls_clean.__len__()*args.batch_size))

        # Processor:
        # ["loss_train", "PER_train", "loss_val", "PER_val"]

        from models.FuncNet1 import FuncNet1
        from models.FuncNet2 import FuncNet2
        from models.FuncNet3 import FuncNet3
        from models.FuncNet4 import FuncNet4
        from models.FuncNet5 import FuncNet5
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
        from models.ConvNet16 import ConvNet16
        from models.DNet1 import DNet1
        from models.DNet2 import DNet2
        from models.RawNet2 import RawNet2

        model_num = [1]

        for i_model, model in enumerate([FuncNet1(**model_kwargs)]):

            model_name = "FuncNet" + str(model_num[i_model])
            visdom_logger_train_ls = VisdomLogger(id + ": " + model_name + " GRU,f=120",
                                                  ["loss_train", "PER_train", "loss_val", "PER_val"], 7)
            processor_ls = InstructionsProcessor(model, training_dataloader, validation_dataloader,
                                                 training_dataloader_ordered=training_dataloader_ordered,
                                                 visdom_logger_train=visdom_logger_train_ls, **processor_kwargs)

            # processor_ls.test_learning_rate_lambda(learning_rate_mode=args.learning_rate_mode, max_lr=args.learning_rate,
            #                                       factor=args.min_learning_rate_factor,
            #                                       step_size_factor=args.learning_rate_step_size, test_epochs=10)
            # processor_ls.find_learning_rate(1e-7, 1e-1, 2)
            processor_ls.print_cuda_information(use_cuda, device)
            print("--------Calling train_model()")

            # processor_ls.load_model("./trained_models/LS_ConvNet" + str(model_num[i_model]) + ".pt")
            # processor_ls.load_model("./trained_models/LS_460ConvNet2.pt")
            # processor_ls.load_model("./trained_models/checkpoint.pt")
            processor_ls.train_model(args.mini_epoch_validation_partition_size,
                                     args.mini_epoch_evaluate_validation,
                                     args.mini_epoch_early_stopping, ordered=True, verbose=False)
            processor_ls.train_model(args.mini_epoch_validation_partition_size,
                                     args.mini_epoch_evaluate_validation,
                                     args.mini_epoch_early_stopping, ordered=False, verbose=False)
            processor_ls.load_model("./trained_models/checkpoint.pt")
            processor_ls.save_model("./trained_models/" + id + "_" + model_name + ".pt")
            visdom_logger_train_ls.save_data_to_file("./logger/" + model_name + "_" + id + ".csv")
            print("Evaluating on test data for " + model_name + ":")
            processor_ls.load_model("./trained_models/checkpoint.pt")
            processor_ls.evaluate_model(testing_dataloader_ls_clean, use_early_stopping=False, verbose=True)
            processor_ls.evaluate_model(testing_dataloader_ls_other, use_early_stopping=False, verbose=True)
            #processor_ls.evaluate_model(testing_dataloader_cv, use_early_stopping=False, verbose=True)
            early_stopper.reset()

        if use_cuda:
            print('Maximum GPU memory occupied by tensors:', torch.cuda.max_memory_allocated(device=None) / 1e9, 'GB')
            print('Maximum GPU memory managed by the caching allocator: ',
                  torch.cuda.max_memory_cached(device=None) / 1e9, 'GB')

    # Todo: Attempt learning completely without CNN, take spec as input to rnn where input_size = num_features.

    # Todo: Move back to tensorboard

# https://github.com/weedwind/CTC-speech-recognition/blob/master/set_model_ctc.py