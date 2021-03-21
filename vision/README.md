# Vision paper

## 1. Model
## 1-1. Classifier model

1. LeNet (1998)
2. AlexNet (2012) : [paper-with-code](https://paperswithcode.com/method/alexnet)
3. VGG (2014) : [paper-with-code](https://paperswithcode.com/method/vgg)
4. GoogLeNet (2014) : [paper-with-code](https://paperswithcode.com/method/googlenet)

## 1-1'. Classifier model (Question)
3. VGG (2014)  
  * Question 1
    parameters : 143,667,240  
    nn.Linear(Fully connected layer) 층에서 약 120,000,000개의 파라미터가 생성됨.  
    이 파라미터의 개수를 추후에 다른 모델에서 어떤 방식으로 감소시킬까?   
    추후 적용하는 'Global Average Pooling' 은 단순 Flatten 하는 것에 비해서 파라미터와 성능 관점에서 어떤, 어느정도의 영향을 미칠까?  
    
    * Global Average Pooling <- Network In Network : [paper-with-code](https://paperswithcode.com/paper/network-in-network#code)  
      (GoogLeNet은 Network-in-Network 논문의 영향을 많이 받았다.)  
      기존 : FC layer 는 overfitting 되기가 쉬워서 Dropout 으로 규제  
      논문 : Global average pooling 을 제안  
           장점 1 : FC layer 보다, GAP layer 가 분류 카테고리에 더욱 유사도, 연관도가 높은 Layer 로서의 역할을 한다.  
           장점 2 : 파라미터가 없어서 이 layer 에서는 overfitting 을 피할 수 있다. (규제로서의 역할) 
      
      


## 1-2. Segmentation model

## 1-3. Object Detection model

