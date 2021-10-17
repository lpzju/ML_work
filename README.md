# ML_work
Learn and study machine learning
Aim: different parameters to see the loss and accuracy

### First attempt：
    - The finally loss is 1.302
    - The accuracy rate is 53%
    - The highest recognition rate is frog with 82% accuracy
    - The lowest recognition rate is dogs with 30% accuracy
    
### Modify rgb channel value：(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    - The finally loss is 1.281
    - The accuracy rate is 53%
    - The highest recognition rate is car with 79% accuracy
    - The lowest recognition rate is cat with 19% accuracy

### Revise batch-size：batch_size = 2
    - The finally loss is 1.302
    - small-batch waste too much time,and didn't perform well
    
### Something about the parameters of CNN
    - Too much parameters to choose/Black box model
    - After checking the information，feel that the parameters given are good enough
    
### Increase the number of epochs：for epoch in range(4)
    - The finally loss is 1.115
    - The accuracy rate is 58%
    - The highest recognition rate is frog with 85% accuracy
    - The lowest recognition rate is dogs with 38% accuracy
    - The recognition accuracy of 7 categories is above 50%

### optimization selection：Adam
    - The finally loss is 1.236
    - The accuracy rate is 57%
    - The highest recognition rate is car with 73% accuracy
    - The lowest recognition rate is cat with 34% accuracy
    - Similar to SGDM algorithm results
