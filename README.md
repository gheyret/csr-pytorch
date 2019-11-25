# csr-pytorch
 Continuous speech recognition using pytorch

Phoneme dictionary used to fill blanks:  
http://www.speech.cs.cmu.edu/cgi-bin/cmudict

# GSC vs Generated
Quick profiling, early stopping patience = 1. i.e this table serves as an indication
but the best results could probably improve.  

          PER                       GSC-test   GEN-test      
          1x Gen data 22050hz:      0.9861     0.0000    gen22050.pt
          5x Gen data 22050hz:      0.9175     0.0000    gen22050x5.pt   
          1x Gen data 16khz:        0.9305     0.0081    gen16000.pt 
          comb training 22050hz:    0.0623     0.0000    mix.pt
          2w mix voice 22050hz:     1.0022     0.0006    gen_2w.pt   
          1x gen 2w:                0.9779     0.0000         
          1x GSC data 22050hz:      0.0543     0.1897    gsc22050.pt   
          1x GSC data 16khz:        0.0594     0.3504    gsc16000.pt
          
  
Gen data: Mix of 3 voices, increased ranges of parameter values. 0.5x amount of GSC. Half of samples are single words, other half are two words.  

          Trained on GSC 22khz:     0.0451     0.5294
                                    0.0462     0.5312
                                    0.0436     0.5419
                                    
          Trained on both 22khz:    0.0565     0.0022
                                    0.0598     0.0036
                                    0.0601     0.0030
                                    
          Trained on partial 50%    0.0602     0.6056
          gsc 22khz
          
          Trained on partial 50%    0.0684     0.0015
          gsc 22khz and generated
          with mix voices 1-2w
          validated only on gsc
          
          Trained on partial 50%    0.0729     0.0036
          gsc 22khz and generated
          with mix voices 1-2w
          validated on comb

          Trained on partial 50%    0.0693     0.0005
          gsc 22khz and generated
          with hazel 1w
          validated on comb
          
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
#Librosa:  
librosa.display.waveplot(wave, sr=SAMPLING_RATE)  
https://www.endpoint.com/blog/2019/01/08/speech-recognition-with-tensorflow  
---

#**Datasets:**
* Google speech commands (currently used) "../data/"
* Generated dataset  (not yet implemented)  
Common voice:  
https://voice.mozilla.org/en/datasets  
LibriSpeech:  
https://www.openslr.org/12/  
VoxForge:  
http://voxforge.org/  
List:  
https://github.com/robmsmt/ASR_Audio_Data_Links  


GSC:  
https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md  
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
