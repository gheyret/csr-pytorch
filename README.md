# csr-pytorch
 Continuous speech recognition using pytorch

Google Speech Commands:
 wav -> mel filterbanks -> equally sized spectrograms -> CNN -> 35 output classes

 
 Utilizes several workers to load data.
 Performs training on GPU.
 All data is preprocessed and loaded in as tensors.
 Cap on performance is disk loading speed.
 Accuracy on validation set 92% reached.
