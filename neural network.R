concrete=read.csv("concrete.csv")
str(concrete)
#Artificial Neural Network best work when input data range should be around zero.
#Now normalization and standardization the input data.
normalize=function(x){
  return((x-min(x))/(max(x)-min(x)))
}
#Now we apply normailze function to all input data.
concrete_norm=as.data.frame(lapply(concrete,normalize))
#To confirm our data is normalized use summary() function.
summary(concrete_norm)#we can see in the data the range between 0 to 1.
concrete_train=concrete_norm[1:773,]
concrete_test=concrete_norm[774:1030,]
#install.packages("neuralnet")
library(neuralnet)
set.seed(100)
concrete_model=neuralnet(strength ~ cement + slag + ash + water+ superplastic +
                                        coarseagg + fineagg + age,
                                            data = concrete_train)
#We can then visualize the network using the plot() function on the
#resulting model:
plot(concrete_model)
model_predict=compute(concrete_model,concrete_test[1:8])
predicted_strength=model_predict$net.result
cor(predicted_strength,concrete_test$strength)
#Correlations close to 1 indicate strong linear relationships between two variables.
#Therefore, the correlation here of about 0.721 indicates a fairly strong relationship.
#This implies that our model is doing a fairly good job, even with only a single hidden node.
#we can improve the performance of our model:
set.seed(123)
concrete_model2=neuralnet(strength~cement + slag + ash + water + superplastic +
                            coarseagg + fineagg + age,data = concrete_norm,hidden = 5)
plot(concrete_model2)
model_predict2=compute(concrete_model2,concrete_test[1:8])
predicted_strength2=model_predict2$net.result
cor(predicted_strength2,concrete_test$strength)
#Notice that the reported error (measured again by SSE) has been reduced from
#5.66 in the previous model to 2.68  here. Additionally, the number of training steps
#from 1431 to 9905, which should come as no surprise given how much more
#complex the model has become. More complex networks take many more iterations
#to find the optimal weights.
#Applying the same steps to compare the predicted values to the true values, we
#now obtain a correlation around 0.901, which is a considerable improvement over
#the previous result of 0.721 with a single hidden node.
