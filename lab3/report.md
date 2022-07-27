# <center> Lab3  Diabetic Retinopathy Detection </center>  

### <center>311554046 林愉修</center>  


## 1. Introduction  

分別使用 ResNet18 以及 ResNet50 來做 transfer learning (用pretrained weights) 和直接對 training dataset 做 learning (Random initialized weights)，任務是對視網膜的照片去做分類，有 0 ~ 4 共 5 種類別，代表視網膜病變的嚴重程度，數字越大代表越嚴重，並且比較兩種 model 以及有沒有用 pretrained weights 的差異。

我的程式設計為先跑 pretrained 再跑沒有 pretrained 的 model，並可以利用 [argparse][15] 來傳入 ResNet18、50的選擇以及超參數等等。

![argparser](https://i.imgur.com/gpn5Osx.png)  


## 2. Experiment setups  

### A. The details of your model (ResNet)  
使用 [torchvision.models][1] 內的 [resnet18][2] 以及 [resnet50][3]，將 fully-connected layer 的 output_feature 設為 5 ，並且 pretrained weights 分別使用 [ResNet18_Weights.IMAGENET1K_V1][4] 以及 [ResNet50_Weights.IMAGENET1K_V2][5]，而 random initialized weights model 則直接將 weights 設為 ```None``` 即可。
> Transfer learning 可分為以下兩種：
> * Feature extraction
將 pretrained model 前面幾層所有的 parameters 皆 freeze 住，只訓練並更新最後一個 layer 的 parameters。
> * Fine-tuning
將 pretrained model 的所有 parameters 都隨著訓練做更新(也有人說是有 unfreeze 一些 layer 的 parameters) 。

我將 pretrained model 設計為先做 feature extraction，再做 fine-tuning，因為聽說這樣比較快達到收斂，並且feature extraction 的 epoch 數可以自己設定。

* ResNet class  
為 [torch.nn.Module][9] 的 subclass，並且可以設定 output_feature 個數，使用 ResNet18 或是 ResNet50 等等。
![ResNet class](https://i.imgur.com/0PDzXwa.png)  

### B. The details of your Dataloader  
RetinopathyLoader 為[torch.utils.data.Dataset][6] 這個 abstract class 的 subclass，並實作其 function，其中我將 transforms 作為instance variable，來存放 [torchvision.transforms][7] 的 function，並且在``` __getitem__ ```先利用 [PIL.Image.open][8] 讀入照片，再將照片經過 transforms 處理，接著抓取該照片的對應 label。
![Retinopathyloader](https://i.imgur.com/U1La6RO.png)  
實作好 Dataset 後，將其放入 [torch.utils.data.DataLoader][10] 以 mini batch 來取得 input 及 label。
![DataLoader](https://i.imgur.com/45HX33Y.png)  
### C. Describing your evaluation through the confusion matrix  
我的 Confusion Matrix 是以 row (True label) 去做 Normalize 的。

* ResNet18 (w/o pretraining) Confusion Matrix  

![ResNet18 (w/o pretraining) Confusion Matrix](https://i.imgur.com/gqiiAqn.png)  

* ResNet50 (w/o pretraining) Confusion Matrix  

![ResNet50 (w/o pretraining) Confusion Matrix](https://i.imgur.com/QEklBd1.png)  

藉由上面兩張圖我們可以清楚看到，沒有 pretraining 的 model，predicted 出來的 label 幾乎都是 0，而在這樣的情況下，Test accuracy 都可以達到 __73%__ 左右，所以我就去看了一下 training 跟 test 的 ground-truth label，發現其中 label 是 0 的比例在 training 跟 test data 也都大約是 __73%__ 左右，是非常 imbalance 的 data，只要全部猜 0 的話，準確率都可以達到 73%，也難怪沒有 pretrained 的 model 直接去對 training data 做訓練會得到像這樣 prediction 幾乎都是 0 的情況。

* ResNet18 (with pretraining) Confusion Matrix  

![ResNet18 (with pretraining) Confusion Matrix](https://i.imgur.com/nCh8LmV.png)  

* ResNet50 (with pretraining) Confusion Matrix  

![ResNet50 (with pretraining) Confusion Matrix](https://i.imgur.com/gzbdacM.png)  
藉由上面兩張圖，可以觀察到有 pretraining 的 model，似乎就稍微改善了這樣的狀況。

不過在 True label 為 1 的情況下，model 還是幾乎都 predict 為 0，並且 predict 2 的數量還比 1 多，同樣在 True label 為 3，也是 predict 為 2 的數量較多，因此我又去看了一下 train 跟 test 的 ground-truth label，果然 label 為 2 的數量又是遠遠多於除了 0 以外的其他 label，加上 1 跟 3 的照片可能分別也會跟 0、2 和 2、4 較為相似，才會有這樣的結果。  


## 3. Experimental results  

### A. The highest testing accuracy  
* Screenshot  
使用 ```ResNet50```，batch size 設為 ```8```，learning rate 設為```1e-3```，momentum 設為 ```0.9```，並且 weight decay 設為 ```5e-4```。
前 5 個 epoch 先做 feature extraction，後 15 個 epoch 做 fine-tuning。
如下圖，對 Data 做 transform
![my data transforms](https://i.imgur.com/3jim4TA.png)  


  得到最高的 Test accuracy 為 __82.36%__
  
* Anything you want to present  
在做 data transform 時，我有先試過使用 pretrained model 所提供的 transforms ([ResNet18_Weights.IMAGENET1K_V1.transforms][11])，如下圖
![pretrained model transforms](https://i.imgur.com/GnPzRPi.png)  
發現使用這樣的 data transform 會造成 pretrained model overfitting，或許跟沒有 data augmentation 有關，因此我將其加入 [torchvision.transforms.RandomHorizontalFlip][12] 以及 [torchvision.transforms.RandomVerticalFlip][13] 雖然改善了overfitting 的問題，但是 testing accuracy 只有大概 __79%__ 左右，後來發現原來是因為有先做 crop 的關係，可能因此造成 crop 到的部分是沒有明顯特徵的，並且在 ResNet 的 fully-connected layer 前有一個 [torch.nn.AdaptiveAvgPool2d][14] 可以確保 output size 都是一致的，因此我就將 crop 以及 resize 去掉，結果testing accuracy 就可以達到 __82%__ 左右。

### B. Comparison figures  
* Plotting the comparison figures  
    * ResNet18  
    ![ResNet18 comparison figure](https://i.imgur.com/niBcJ5m.png)  
    
    * ResNet50  
    ![ResNet50 comparison figure](https://i.imgur.com/gaatL3w.png)  


## 4. Discussion  

### A. Anything you want to share  
在實驗過程中有發現到 imbalanced data 對於 model 所帶來的影響，我也去了解該如何改善它，發現到可以使用以下幾種方法：
1. Oversampling  
把 minority class 補到跟 majority class 數量一樣多，可以透過 data aumentation 的方式，但有可能會造成 overfitting。
2. Undersampling  
把 majority class 砍到跟 minority class 數量一樣多，透過隨機刪除的方式，但有可能刪到一些重要的 feature，造成 underfitting。
3. Class-weighted loss  
對 minority class 的 loss 乘上一個較大的 weight，讓它對於 loss 的影響較多，因為我們的目標是要最小化 loss ，所以就會特別去學習 minority class，常見決定 class weight 的方式是 
    * $$weight_{類別} = \frac{1}{類別數量}$$  
    * $$weight_{類別} = 1 - \frac{該類別數量}{總資料量}$$  
    * $$weight_{類別} = \frac{最多類別資料量}{該類別資料量}$$  

這邊我以上述第二種方式來決定 weight ，```weight = [0.2649, 0.9304, 0.8502, 0.9752, 0.9793]```，並以此 weighted loss 再訓練一次 ResNet50，得到以下結果。
* ResNet50 (with pretraining & weighted loss) Confusion Matrix  
![ResNet50 (with pretraining & weighted loss) Confusion Matrix](https://i.imgur.com/xOvMkrQ.png)  

可以看到相較於沒有 weighted loss 時，Confusion Matrix 有改善一些，prediction 為 0 跟 2 的比例減少，而 1 跟 3 則增加，特別在 True label 為 3 的情況下，Predict label 為 3 的比例也大幅增加，不過在 True label 為 1 的情況下，Predict 為 1 的比例雖然有增加，但是只有增加了一點點。

* ResNet50 (with weighted loss) Comparison Figure  



[1]: https://pytorch.org/vision/stable/models.html "https://pytorch.org/vision/"
[2]: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18 "https://pytorch.org/vision/"
[3]: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50 "https://pytorch.org/vision/"
[4]: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights "https://pytorch.org/vision/"
[5]: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights "https://pytorch.org/vision/"
[6]: https://pytorch.org/docs/1.12/data.html?highlight=dataset#torch.utils.data.Dataset "https://pytorch.org/docs/"
[7]: https://pytorch.org/vision/0.13/transforms.html "https://pytorch.org/vision/"
[8]: https://pillow.readthedocs.io/en/stable/reference/Image.html#module-PIL.Image "https://pillow.readthedocs.io/"
[9]: https://pytorch.org/docs/1.12/generated/torch.nn.Module.html#torch.nn.Module "https://pytorch.org/docs/"
[10]: https://pytorch.org/docs/1.12/data.html?highlight=dataloader#torch.utils.data.DataLoader "https://pytorch.org/docs"
[11]: https://pytorch.org/vision/stable/models.html "https://pytorch.org/vision/"
[12]: https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip "https://pytorch.org/vision/"
[13]: https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomVerticalFlip.html#torchvision.transforms.RandomVerticalFlip "https://pytorch.org/vision/"
[14]: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html "https://pytorch.org/docs/"
[15]: https://docs.python.org/3/library/argparse.html "https://docs.python.org/"