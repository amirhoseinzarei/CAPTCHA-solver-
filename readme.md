
* To run this please just go to your terminal and type:
#
>`pip install runipy `

Then

>`runipy ocr.ipynb`
#

and if it does not work you can try 

>`ipython ocr.ipynb`




#



**OCR MODEL FOR READING CAPTCHAS**

This project has 4 files :

*1 - ocr.ipynb* -> The main file.

*2 - utils.py* -> This file provide datasets , preprocessing them and help to have cleam code.

*3 - CTCLayer.py* -> This is the CTC layer and it used for model.

*4 - SquareRootScheduler.py* -> This is Learning Rate Scheduler for change Learning Rate during training process.

* first in utils.py we extract our images and labels of them
then by 3 functions preprocessing them such as normalization and...  also provide them to create training and validation dataset objects to get model

* then in ocr.ipynb after import required library and modules we set parameters like number of epoch , batch size and ... 

    after it , in build_model function we put 2  convolutional blocks to extract a sequence of features  then we have to reshape out put of secend  convolutional block to get it to 3 bidirectional GRU layers to propagate information through this sequence.

    we use bidirectional GRU because a typical state in an RNN (simple RNN, GRU, or LSTM) relies on the past and the present events. A state at time t depends on the states x1,x2,…,xt−1, and xt . However, there can be situations where a prediction depends on the past, present, and future events. future mean reverse traversal of input.

    for a detailed guide to bidirectional please check out this link : https://github.com/christianversloot/machine-learning-articles/blob/main/bidirectional-lstms-with-tensorflow-and-keras.md
    
    in the end of model we instead of use loss function in `model.fit` like usual , we implement custom loss function in last layer that is CTC layer, in other words it is part of the model and not seperate from the model.




 * CTC layer solves 2 important problems for us.

     for a detailed guide to CTCLayer please check out this link:
 https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c  

 
* after that we create callback to control training process with early stopping and learning rate scheduler.

also for a detailed guide to Learning Rate Scheduler please check out this link :
https://towardsdatascience.com/learning-rate-scheduler-d8a55747dd90

 at the end of the work fit model and plot the results.
