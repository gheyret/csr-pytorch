# csr-pytorch
 Continuous speech recognition using pytorch

---

##Pipeline:

* Reads wav form from hdf5  
* Transforms it into a spectrogram
* Input spectrogram to CNN to give probability distribution of phoneme classes including blank
* The probability distribution is input to CTC for decoding and computing loss compared to true label sequence

---

##**Datasets:**
* Google speech commands (currently used) "../data/"
* Generated dataset  (not yet implemented)

These require different dataloaders since the conversion from word to phoneme sequence is hard coded currently


---
##TODO:
**Take care of random scripts.**


**CTC Decoding:**  
[ ] Greedy  
[ ] Beam search  
[ ] CTC_Loss ?  
[ ] LER/Edit distance??  
https://github.com/gaoyiyeah/speech-ctc/blob/master/speech/models/ctc_decoder.py  
https://github.com/jpuigcerver/pytorch-baidu-ctc  
https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py  
https://github.com/ottokart/beam_search/blob/master/beam_search.py  
https://github.com/parlance/ctcdecode  
https://medium.com/@kushagrabh13/modeling-sequences-with-ctc-part-1-91b14a0405b3  
https://medium.com/@kushagrabh13/modeling-sequences-with-ctc-part-2-14ab45ef896e  