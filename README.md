# Reproduction of ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation


## Introduction
In this project, we managed to reproduce the result of table 1(a) and did data migration on a different dataset `foggy Cityscape`. 
In the paper of ADVENT, the authors proposed two novels, complementary methods using (i) a direct entropy loss and (ii) an adversarial loss respectively. They demonstrated state-of-the-art performance in semantic segmentation on two challenging “synthetic-2-real” set-ups1 and show that the approach can also be used for detection.
## Main approach
Two approaches are implemented targeting minimizing the entropy loss, including direct entropy minimization and adversarial training minimization.
<!-- ![Architecture](https://i.imgur.com/zOCWrxN.png) -->

<img src="https://i.imgur.com/zOCWrxN.png"/>
<center><p>Figure 1. Architecture</p></center>

### Direct entropy minimization
The key idea for this method is to approximate true y_t distribution with the model prediction as an unsupervised approach. Rather than choosing approximations with high confidence, constraints are proposed to force the model to always produce high-confident predictions, which is done by minimizing the entropy loss.
The entropy loss L_ent is defined as the sum of all pixel-wise normalized entropies, and the optimization is implemented in a joint form of supervised segmentation loss and unsupervised entropy loss Lent, represented by the following formula, in which Shannon Entropy[^5] was used as L_ent:

<img align=center width="460" alt="Screenshot 2022-04-14 at 20 06 54" src="https://user-images.githubusercontent.com/49361681/163451070-c483abb9-db76-4350-9afa-e2913921d5e1.png">


### Adversarial training
For this approach, a fully-convolutional discriminator network is constructed and then trained to discriminate between source images and generated target images. Meanwhile, the segmentation network is trained to fool the discriminator. The optimization formula can be represented as:
<img width="460" alt="Screenshot 2022-04-14 at 20 11 27" src="https://user-images.githubusercontent.com/49361681/163451482-7969285f-7b5d-4908-a431-944cace36677.png">



## Experiment
### Dataset
The task is defined as unsupervised domain adaption from synthetic to real-world scenarios. In the implementation, GTA5 data is used as the source while the Cityscape dataset is used as the target. The GTA5 dataset consists of 24,966 synthesized frames captured from a video game[^1]. Images are provided with pixel-level semantic annotations of 33 classes. For target domain training, 2,975 unlabeled Cityscape images are used with the standard mean-Intersection-over-Union(more) metric and then tested on 500 validation images[^2].
### Implementation details
All the training and testing procedure are run with pytorch. For the training of discriminator, Adam optimizer[^3] is used with learning rate equals to 0.0001, while in all other training cases SGD optimizer is used with learning rate equals to 0.0025, momentum equals to 0.9 and weight decay of 0.0001.
## Reproduce process
### Hyperparameter tuning
To finish the essential requirement, we first read the code to find the configuration files that give the settings. We manually check the required files to start the process. The author uses the `.yml` file to configure the training and testing files. By modifying those files, we begin to train the baselines. We need to prepare the model using the negative entropy and the minimal entropy to get two different model parameters for this reproduction goal. Observing the table, the other row is based on those baselines. After finishing the first two-line, the others are pretty easy. During the training, we find something that is not proper for ourselves. So we changed some hyperparameters to fit our training process that including:
* Learning Rate
* Weight decay
* Train epochs
* Batch size
* Step size
* Number of GPU

For example, our computers have only one GPU while there are usually more than one for ordinary severs. So, we can't follow the setting of parallel processing. We disabled the setting and checked the hyperparameters of the training, like the learning rate and the decay. We change the speed to study the influence and find out that the initial setting works best compared to the others. As the original folder does not offer the data set they use, we download the dataset from two websites that contain the famous benchmark cityscapes. We look through the data set and find that they use different colors to represent other items. They use the screenshot of the game GTA5 as the training data set since the game is well designed and similar to the real world. We modified the data set size while finding the performance was not that good. Then we follow the instruction of the configuration files, and the results are quite an idea, which means that the essential requirement is fulfilled.
### Dataset migration
Then we looked through the data set again, and we noticed no noise in the training and testing set. So, we wonder what if we import some noise to the test image. We found some helpful image sets offered by the cityscapes. It is called the foggy image. In these figures, the items are sometimes covered by fog. Therefore, the items are not fully shown or could be regarded as adding some noise. We modified the `cityscape.py` and `test.py` to match the file name of the foggy data. After the modification, we run the test on the foggy data; The results indicate that the noise does decrease the accuracy of the network, but it still works. In this way, we think this is a generalized network.
## Results 
|                      | road  | sidewalk | building | wall | fense | pole | light | sign | veg  | terrain | sky  | person | rider | car  | truck | bus  | train | mbike | bike | mIoU |
|----------------------|-------|----------|----------|------|-------|------|-------|------|------|---------|------|--------|-------|------|-------|------|-------|-------|------|------|
| minent               | 76,3  | 16,8     | 66,53    | 32,4 | 18,4  | 18,2 | 20,3  | 7,8  | 75,2 | 24,3    | 62,1 | 40,8   | 4,8   | 79,5 | 14,6  | 8,5  | 0     | 14,5  | 0,4  | 31,5 |
| advent               | 79,4  | 27       | 70,2     | 14,8 | 11,8  | 27   | 26,9  | 18,3 | 80,3 | 29,1    | 71,8 | 54,1   | 17,5  | 78,1 | 11,5  | 14,3 | 0,2   | 9,6   | 0    | 33,8 |
<center><p>Table 1. Result on VGG-16</p></center>

|                      | road  | sidewalk | building | wall | fense | pole | light | sign | veg  | terrain | sky  | person | rider | car  | truck | bus  | train | mbike | bike | mIoU |
|----------------------|-------|----------|----------|------|-------|------|-------|------|------|---------|------|--------|-------|------|-------|------|-------|-------|------|------|
| minent               | 84,4  | 18,7     | 80,6     | 23,8 | 23,2  | 28,4 | 36,9  | 23,4 | 83,2 | 25,2    | 79,4 | 59     | 29,9  | 78,5 | 33,7  | 29,6 | 1,7   | 29,9  | 33,6 | 42,3 |
| minent+ER            | 84,2  | 25,2     | 77       | 17   | 23,3  | 24,2 | 33,3  | 26,4 | 80,7 | 32,1    | 78,7 | 57,5   | 30    | 77   | 37,9  | 44,3 | 1,8   | 31,4  | 36,9 | 43,1 |
| advent | 89,9 | 36,5 | 81,6 | 29,2 | 25,2 | 28,5 | 32,3 | 22,4 | 83,9 | 34 | 77,1 | 57,4 | 27,9  | 83,7 | 29,4  | 39,1 | 1,5   | 28,4  | 23,3 | 43,8 |
| advent+minent        | 89,4  | 33,1     | 81       | 26,6 | 26,8  | 27,2 | 33,5  | 24,7 | 83,9 | 36,7    | 78,8 | 58,7   | 30,5  | 84,8 | 38,5  | 44,5 | 1,7   | 31,6  | 32,4 | 45,5 |
<center><p>Table 2. Result on ResNet-101</p></center>

In the reproduction experiment, we mainly focused on reproducing "Table 1(a) ours" of the paper. The mean Intersection Over Union is a typical metric to evaluate the segmentation result. We found that using the existing code and parameters mentioned in the paper, we reproduced some close results on GTA to Cityscape. In table 1, we reproduced the result using the VGG-16-based model. In table 2, the results are reproduced using the ResNet-101-based model. The result shows that the Res-net model has better performance on multiple classes. By combining the results of the two models MinEnt and Ad-vEnt, we observe a decent boost in performance, compared to the results of single models. Such a result indicates that complementary information is learned by the two models.

Among the 17 classes, some objects are well segmented, while others' segmentation is less effective. We assume that these objects are common in the images and count for a large proportion of the image such as roads, and cars.

Although the reproduction result we obtained is close to the given table, there are still outliers. In our result lt, the next IOU of the 'like eminent' class is 0, but it is 18.9 in the paper. We noticed that in other approaches(combined models and different single el), the IOU of model classlass momodellass is also close to 1.2. In this paper, the authors did not explicitly explore the reason behind it, we think that this outlier is caused by  misclassification of other classes.

| | road | sidewalk | building | wall | fense | pole | light | sign | veg | terrain | sky | person | rider | car | truck | bus | train | mbike | bike | mIoU |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| minent(foggy) | 82,4 | 29,8 | 76,3 | 14,6 | 22 | 19,5 | 32 | 11,9 | 82,8 | 29,5 | 76,7 | 56 | 27,3 | 77 | 33,3 | 12,5 | 1,7 | 25,5 | 31,4 | 41,2 |
| advent(foggy) | 84,6 | 30,7 | 78,7 | 25,4 | 22,1 | 24,8 | 30 | 18,8 | 80,7 | 27,1 | 55,1 | 57,4 | 26,3 | 85 | 25,9 | 31,7 | 1,2 | 30,5 | 13   | 40,6 |
| advent+minent(foggy) | 85,2 | 22,3 | 61,3 | 12,2 | 15,8 | 15,3 | 18    | 15,4 | 45,9 | 26,6 | 54,5 | 47,4 | 26,4 | 77,6 | 15 | 19,8 | 1,1 | 28,1 | 22,7 | 32,1 |
<center><p>Table 3. Result on foggy data</p></center>

In the paper, the authors claimed that the model,tabl e training can also be used in unsupervised domain adaptation. So we also exploeminente meminent advent, and their combined performance on the `foggy Cityscape` dataset which is a dataset generated using depth information[^4]. Compared to the baseline, the single model performed very well in almost every class. But we also found that the combination model did not perform very well in some classes. It seems the authors did not mention this situation, but we assume that the combination model overfits the source domain and thushase a very unsatisfactory result on the foggy target domain. 

We also find some interesting outliers. In the paper, when using the VGG-16 model in the able, the ttrainingraitraining rainn class of advent entropy is 0.4% with other methods and networks. But our reproduction result shows that it is close to other results which are close to 1% in the table, so we think this is an example of overfitting on some images.  

## Conclusion
In this reproduction project, we read the paper to understand the main approach. The main contribution of this paper is introducing the adversarial loss function to the CNN network. We apply two different networks for the loss function and combine two loss functions to form a new network. It shows that the combined networks perform best in the sunny data set. We also apply the foggy data set and found that the baseline works well although there are some accuracy declines. However, the combined network is not that good which we thought was caused by the overfitting. Also, we observed an unexpected accuracy for a train in the advent models. It is interesting and we can not find a proper explanation for it and the author did not address it too. The reproduction is quite meaningful and helps us recognize the whole process to form research.

## Contributions

### Jingyu Li
In this project, I did some research on the dataset and domain in the paper. In that the paper focused on domain adaptation, besides the required table 1 results, I explored the possibility to adapt the model to other domains. I found some interesting research datasets and did data migration on the foggy Cityscape domain. Together with Nianru, We can write a new class for the foggy Cityscape dataset and got reproduction results. We found some interesting improvements in the baseline model and clear signs of overfitting happened to the combination of the 2 proposed approaches.
### Nianru Wang
For this project, I was mainly responsible to the reproduce. I borrowed a computer sever that has cuda and nice cpu. Compared to our personal computer, it is much more effciency than ours. Then I modified the code to fit for the sever and change the configuration files. This project use `.yml` files to set the hyperparameters. I change the hyperparameter like the learning rate and weight decay. I find that the default setting is optimal.
### Xuanyu Zhuang
My task was to figure out the principle and mathematics behind the approaches presented in the paper, as well as the definitioin of training datails and the choice of optimizers and networks. During the training process, I tuned the parameters including weight decay and learning rate together with Nianru to see if there is possible better result, however the result presented in the paper turned out to be already optimal. Finally, I shared some work of PPT preparation and blog writing.
### Zhiheng Wang
In this task,my job was to cooperate with Nianru for reproduce of this paper.During the traning process,I've tried to figure out the interactions between different cases based on parameter tuning.The final result of our reproduce work match with the evaluation result in the original paper.Lastly,I made the poster of our reproduce job,which demonstrate the kernal idea of this paper.

[^1]: https://arxiv.org/abs/1608.02192
[^2]: https://www.cityscapes-dataset.com/dataset-overview/
[^3]: https://arxiv.org/abs/1412.
[^4]: http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/
[^5]: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
