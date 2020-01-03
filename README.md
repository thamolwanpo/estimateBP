# Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network

This respository was created to implement this paper which title is   
'[Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8037207)' by Frank P.-W. Lo, Charles X.-T. Li, Jiankun Wang, Jiyu Cheng and Max Q.-H. Meng, Fellow, IEEE

The network model of this method is seq2seq which I developed using Keras. (for more information please visit [link](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction))

## Dataset
You can download the dataset [here](https://drive.google.com/file/d/1veMt3YrkE17bivOYrrbLob2r7yRLJmyc/view?usp=sharing). 
After Downloaded the dataset put it in ```data/```.

## Test on
- python 3.7.4
- keras 2.3.1
- tensorflow 2.0.0
- numpy 1.17.2

## Run the model
To run the code please open jupyter notebook:
1. Run ```preprocess.ipynb```
2. Run ```train.ipynb```

## Experimental Results
The results will be different from the paper because I used 84 subjects, so the result will be a little bit similar to  
[Long-term Blood Pressure Prediction with Deep Recurrent Neural Networks](https://arxiv.org/abs/1705.04524) which also used 84 subjects.
<img src="https://github.com/ploymel/estimateBP/blob/master/pics/table.png" width="300">

My Results are:
```
Overall RMSE Systolic: 6.053 (mmHg)
Overall RMSE Diastolic: 3.402 (mmHg)
```

## For future modification to Improve Model Performance
You can modify the network model in ```model.py``` and test it in ```model_eval.ipynb```

## References
- [Continuous Systolic and Diastolic Blood Pressure Estimation Utilizing Long Short-term Memory Network](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8037207) 
by Frank P.-W. Lo, Charles X.-T. Li, Jiankun Wang, Jiyu Cheng and Max Q.-H. Meng, Fellow, IEEE
- [Keras implementation of a sequence to sequence model for time series prediction using an encoder-decoder architecture.](https://github.com/LukeTonin/keras-seq-2-seq-signal-prediction)
by Luke Tonin
- [Long-term Blood Pressure Prediction with Deep Recurrent Neural Networks](https://arxiv.org/abs/1705.04524)
by Peng Su, Xiao-Rong Ding, Yuan-Ting Zhang, Jing Liu, Fen Miao, Ni Zhao
