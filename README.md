# Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network

This respository was created to implement this paper which title is   
'[Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8037207)'

The network model of this method is seq2seq which I developed using Keras. (for more information please visit [link](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction))

## Run the model
To run the code please open jupyter notebook:
1. Run ```preprocess.ipynb```
2. Run ```train.ipynb```

## For future modification to Improve Model Performance
You can modify the network model in ```model.py``` and test its in ```model_eval.ipynb```

## References
- [Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8037207) 
by Frank P.-W. Lo, Charles X.-T. Li, Jiankun Wang, Jiyu Cheng and Max Q.-H. Meng, Fellow, IEEE
- [Keras implementation of a sequence to sequence model for time series prediction using an encoder-decoder architecture.](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction)
by Luke Tonin
