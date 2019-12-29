import_kwargs = {"vocabulary_path": args.vocab_path, "vocabulary_path_is_xml": args.vocab_is_xml,
                 "vocabulary_path_addition": args.vocab_addition_path,
                 "label_type": args.label_type}
dataset_kwargs = {"num_features_input": args.num_features_input,
                  "use_delta_features": args.use_delta_features, "input_type": args.input_type}
model_kwargs = {"num_classes": get_num_classes(args.label_type), "num_features_input": args.num_features_input,
                "num_input_channels": num_input_channels, "non_linearity": args.non_linearity,
                "rnn_bidirectional": args.rnn_bidirectional,
                "memory_type": args.rnn_memory_type, "input_type": args.input_type}



libri_speech_test_clean_path = args.libri_path + "test-clean/"
libri_speech_test_other_path = args.libri_path + "test-other/"

collection_ls_test_clean, missing_words_ls_test_clean = import_data_libri_speech(
    dataset_path=libri_speech_test_clean_path, **import_kwargs)
collection_ls_test_other, missing_words_ls_test_other = import_data_libri_speech(
    dataset_path=libri_speech_test_other_path, **import_kwargs)

testing_set_ls_clean = Dataset(collection=collection_ls_test_clean, **dataset_kwargs)
testing_dataloader_ls_clean = AudioDataLoader(testing_set_ls_clean, **dataloader_kwargs)

testing_set_ls_other = Dataset(collection=collection_ls_test_other, **dataset_kwargs)
testing_dataloader_ls_other = AudioDataLoader(testing_set_ls_other, **dataloader_kwargs)

