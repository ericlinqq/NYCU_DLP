Lab2 EEG Classification
===
### 311554046 林愉修
## 1. Introduction
利用 Pytorch implement EEGNet 和 DeepConvNet來做 Binary Classification，資料為 BCI Competition III - Dataset IIIb，資料的shape為(2, 750)。
## 2. Experiment set up
A. The details of your model
* EEGNet
![EEGNet](https://i.imgur.com/6CWHPsH.png)
此 model 使用了 Depthwise seperable Convolution，目的是希望在不影響輸出結構的情況下去減少運算量，EEGnet 由1個普通 conv + 1個Depthwise conv + 1個 Separable conv 組成，其中，Separable Convolution 由一個 Depthwise Convolution 和一個 Pointwise Convolution 組成。

>Depthwise convolution:
    和一般 convoution 不同，會建立與 input channel 數相同個數的filter，並且每個 filter 針對對應的 channel 分開去做 convolution。

>Pointwise convolution:
    先對每個輸出 channel 建立一個大小為 1×1×M 的 filter 後 (M 為輸入層的 channel 數)，將輸入層的所有點進行 convolution 運算。假如輸出層有 N 個 channel，則會建立 N 個 1×1×M 的 filter。

* DeepConvNet
![DeepConvNet](https://i.imgur.com/i4GfShA.png)
傳統的CNN架構，經過多層(Conv, BatchNorm, Activation, MaxPool, Dropout)，並且 Conv 及 MaxPool 沒有做 padding。

B. Explain the activation function(ReLU, Leaky ReLU, ELU)
* ReLU
![ReLU](https://i.imgur.com/mtjwogW.png)
![ReLU graph](https://miro.medium.com/max/714/1*oePAhrm74RNnNEolprmTaQ.png)
由於 Sigmoid 的 Gradient Vanishing 問題，因此衍生出Recified Linear Unit (ReLU)這個activation function，其優點除了減少Gradient Vanishing 問題外，運算也十分簡單，使得計算成本下降，不過它可能會導致權重更新不了，因 input < 0 時，gradient為0，這樣會沒辦法更新weight，稱為dying ReLu problem，也因此後來提出 Leaky ReLU 來改善這個問題。
* Leaky ReLU
![LeakyReLU](https://i.imgur.com/wADzIZ1.png)
![LeakyReLU graph](https://miro.medium.com/max/796/1*FDOyQlRurCK7mWU5i0Ly_w.png)
具有 ReLU 的特點，但是在 input < 0 時，gradient不為0，而是一個很小的數，可以解決dying ReLU problem。
* ELU
![ELU](https://i.imgur.com/ur8ptbG.png)
![ELU graph](https://pytorch.org/docs/stable/_images/ELU.png)
同樣是為了解決dying ReLU problem，不過計算上較Leaky ReLU複雜。
## 3. Experiment results
A. The highest testing accuracy
* Screenshot with two models  
    兩種 model使用同樣的 Hyperparameters，batch size 設為 540，optimizer 使用 Adam，並且 learning rate 設為 0.001，weight decay 設為 0.01，activation function 參數皆使用預設值，跑 300 epochs。  
    
    |  |  ReLU_test | LeakyReLU_test | ELU_test |
    | --- | --- | --- | --- |
    | EEGNet | 87.59% | 85.27% | 84.35% |
    | DeepConvNet | 81.94% | 83.70% | 80.00% |

* Anything you want to present
    一開始 batch size 設為 64，跑出來最好的 test accuracy 大約只有 84% 左右，之後加大 batch size 為 540 後， test accuracy 也跟著上升至 87% 左右，不過再加大至 1080 (full batch) 時，test accuracy 反而會下降至 85% 左右，可能跟大的 batch size 容易陷入 sharp minima 有關。

B. Comparison figures
* EEGNet
![EEGNet Comparison](https://i.imgur.com/1xl01RS.png)

* DeepConvNet
![DeepConvNet Comparison](https://i.imgur.com/QaUfvJq.png)

## 4. Discussion
A. Anything you want to shares
[torch.nn.CrossEntropy()][1] 要求 label 的 datatype 必須為 torch.long。
要將存好的 model weights load 進來時，出現錯誤，原因是我使用gpu 進行訓練，而 load 的裝置只有 cpu，因此須將 [torch.load()][2] 中的 map_location 參數設為 'cpu' 即可解決。

  [1]: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss       "https://pytorch.org/docs"
  [2]: https://pytorch.org/docs/stable/generated/torch.load.html "https://pytorch.org/docs"