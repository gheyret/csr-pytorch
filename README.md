# csr-pytorch
 Continuous speech recognition using pytorch

---
             GSC        Val Loss        Params
ConvNet2 :  3.7%PER     0.177223        14.3M  
3x LSTM  :  6.7%PER     0.296101        18.5M

---

#Pipeline:

* Reads wav form from hdf5  
* Transforms it into a spectrogram
* Input spectrogram to CNN to give probability distribution of phoneme classes including blank
* The probability distribution is input to CTC for decoding and computing loss compared to true label sequence

---
Librosa:  
librosa.display.waveplot(wave, sr=SAMPLING_RATE)  
https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow  
---

#**Datasets:**
* Google speech commands (currently used) "../data/"
* Generated dataset  (not yet implemented)  
https://voice.mozilla.org/en/datasets  
https://www.openslr.org/12/  
http://voxforge.org/  
https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz  

These require different dataloaders since the conversion from word to phoneme sequence is hard coded currently.
It also requires a new model since the sample rate is different.


---
**CTC Decoding:**  
[X] Greedy  
[X] Beam search  
[X] CTC_Loss ?  
[X] LER/Edit distance??  
[ ] Tensor implementation of CTC beam search and CTC loss
https://github.com/gaoyiyeah/speech-ctc/blob/master/speech/models/ctc_decoder.py  
https://github.com/jpuigcerver/pytorch-baidu-ctc  
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py  
https://github.com/ottokart/beam_search/blob/master/beam_search.py  
https://github.com/parlance/ctcdecode  
https://medium.com/@kushagrabh13/modeling-sequences-with-ctc-part-1-91b14a0405b3  
https://medium.com/@kushagrabh13/modeling-sequences-with-ctc-part-2-14ab45ef896e  

---
**Useful Gits**  
https://github.com/Holmeyoung/crnn-pytorch  
https://github.com/LearnedVector/Wav2Letter  
https://github.com/SeanNaren/deepspeech.pytorch  
https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow  
https://github.com/jinserk/pytorch-asr/tree/master/asr  

---
**TensorBoard**  
https://itnext.io/how-to-use-tensorboard-5d82f8654496  
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html  
https://pytorch.org/docs/stable/tensorboard.html  
https://discuss.pytorch.org/t/visualize-live-graph-of-lose-and-accuracy/27267  
https://github.com/facebookresearch/visdom  


---
**RNN:**
* N x 1 x F x T
* -> Conv
* N x FM x F x T

* Combine before RNN ->
* N x (FM*F) x T

* Reorder ->
* N x T x (FM*F)
* T x N x (FM*F)

* Rnn defined such that: FM*F = input_size, hidden_size = some value
* Then add a fully connected layer after that projects hidden_size for each timestep to number of classes.

---
**Profiling**  
-python -m cProfile -o prof.out train_model.py  
-snakeviz prof.out  
