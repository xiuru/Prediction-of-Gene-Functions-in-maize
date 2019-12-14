#' ---
#' title: "caret"
rm(list=ls())
library(caret)

#'
## ------------------------------------------------------------------------
label=read.csv("label.csv",head=F)
str(label)
f = read.csv('Imputed-features.csv',row.names=1)
data = f[match(label[,1],rownames(f)),]
dim(data)
x = data; y = as.factor(label$V2)

ProcValues = preProcess(data, method = "nzv")
nzv = predict(ProcValues,data)
data.final = nzv
dim(data.final)

data.pca <- prcomp(data.final,rank=20,center=TRUE,scale=TRUE)
summary(data.pca)
pncp.comp <- as.data.frame(data.pca$x)
dim(pncp.comp)
pca=pncp.comp

#'
#' ## spliting training and testing data
## ------------------------------------------------------------------------
index <- createDataPartition(label$V2, p=0.8, list=FALSE)
trainSet <- x[index,]
trainSet.y <- y[index]
testSet <- x[-index,]
testSet.y <- y[-index]


#'
#' ## define the fitcontrol
## ------------------------------------------------------------------------
# imp = varImp
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = 'final',
  classProbs = T)

#'
#' ## test different method have same important vactors or not
#' ## random forest
## ------------------------------------------------------------------------
set.seed(123)
model_rf<-train(trainSet,trainSet.y,method='rf',trControl=fitControl,tuneLength=5)
imp = varImp(model_rf)
imp2 <- data.frame(overall = imp$importance[,1],
                  names   = rownames(imp$importance))
confusionMatrix(model_rf$pred$obs,model_rf$pred$pred,positive='Y')

set.seed(123)
model_rf.pca<-train(pca,trainSet.y,method='rf',trControl=fitControl,tuneLength=5)
confusionMatrix(model_rf.pca$pred$obs,model_rf.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_rf2<-train(trainSet[,predictors],trainSet.y,method='rf',trControl=fitControl,tuneLength=5)
confusionMatrix(model_rf2$pred$obs,model_rf2$pred$pred,positive='Y')

write.csv(model_rf$pred,'model_rf.pred.csv')
write.csv(model_rf.pca$pred,'model_rf.pca.pred.csv')
write.csv(model_rf2$pred,'model_rf2.pred.csv')

#'
#' ## nnet
## ------------------------------------------------------------------------
set.seed(123)
model_nnet<-train(trainSet,trainSet.y,method='nnet',trControl=fitControl,tuneLength=5)
imp = varImp(model_nnet)
imp2 <- data.frame(overall = imp$importance[,1],names   = rownames(imp$importance))
confusionMatrix(model_nnet$pred$obs,model_nnet$pred$pred,positive='Y')

set.seed(123)
model_nnet.pca<-train(pca,trainSet.y,method='nnet',trControl=fitControl,tuneLength=5)
confusionMatrix(model_nnet.pca$pred$obs,model_nnet.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_nnet2<-train(trainSet[,predictors],trainSet.y,method='nnet',trControl=fitControl,tuneLength=5)
confusionMatrix(model_nnet2$pred$obs,model_nnet2$pred$pred,positive='Y')

write.csv(model_nnet$pred,'model_nnet.pred.csv')
write.csv(model_nnet.pca$pred,'model_nnet.pca.pred.csv')
write.csv(model_nnet2$pred,'model_nnet2.pred.csv')

#'
#' ## svm
## ------------------------------------------------------------------------
set.seed(123)
model_svm<-train(nzv,trainSet.y,method='svmRadial',trControl=fitControl,tuneLength=5)
imp = varImp(model_svm)
imp2 <- data.frame(overall = imp$importance[,1], names   = rownames(imp$importance))
confusionMatrix(model_svm$pred$obs,model_svm$pred$pred,positive='Y')

set.seed(123)
model_svm.pca<-train(pca,trainSet.y,method='svmRadial',trControl=fitControl,tuneLength=5)
confusionMatrix(model_svm.pca$pred$obs,model_svm.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_svm2<-train(trainSet[,predictors],trainSet.y,method='svmRadial',trControl=fitControl,tuneLength=5)
confusionMatrix(model_svm2$pred$obs,model_svm2$pred$pred,positive='Y')

write.csv(model_svm$pred,'model_svm.pred.csv')
write.csv(model_svm.pca$pred,'model_svm.pca.pred.csv')
write.csv(model_svm2$pred,'model_svm2.pred.csv')

#'
#' ## lasso
## ------------------------------------------------------------------------
set.seed(123)
model_lasso<-train(trainSet,trainSet.y,method='glmnet',trControl=fitControl,tuneLength=5)
imp = varImp(model_lasso)
imp2 <- data.frame(overall = imp$importance[,1],names   = rownames(imp$importance))
confusionMatrix(model_lasso$pred$obs,model_lasso$pred$pred,positive='Y')

set.seed(123)
model_lasso.pca<-train(pca,trainSet.y,method='glmnet',trControl=fitControl,tuneLength=5)
confusionMatrix(model_lasso.pca$pred$obs,model_lasso.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_lasso2<-train(trainSet[,predictors],trainSet.y,method='glmnet',trControl=fitControl,tuneLength=5)
confusionMatrix(model_lasso2$pred$obs,model_lasso2$pred$pred,positive='Y')

write.csv(model_lasso$pred,'model_lasso.pred.csv')
write.csv(model_lasso.pca$pred,'model_lasso.pca.pred.csv')
write.csv(model_lasso2$pred,'model_lasso2.pred.csv')

#'
#' ## qda  error: some group is too small for 'qda'
#'
#' ## lda
## ------------------------------------------------------------------------
set.seed(123)
model_lda<-train(nzv,trainSet.y,method='lda',trControl=fitControl,tuneLength=5)
imp = varImp(model_lda)
imp2 <- data.frame(overall = imp$importance[,1], names= rownames(imp$importance))
confusionMatrix(model_lda$pred$obs,model_lda$pred$pred,positive='Y')

set.seed(123)
model_lda.pca<-train(pca,trainSet.y,method='lda',trControl=fitControl,tuneLength=5)
confusionMatrix(model_lda.pca$pred$obs,model_lda.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_lda2<-train(trainSet[,predictors],trainSet.y,method='lda',trControl=fitControl,tuneLength=5)
confusionMatrix(model_lda2$pred$obs,model_lda2$pred$pred,positive='Y')

write.csv(model_lda$pred,'model_lda.pred.csv')
write.csv(model_lda.pca$pred,'model_lda.pca.pred.csv')
write.csv(model_lda2$pred,'model_lda2.pred.csv')

#'
#' ## MLR
## ------------------------------------------------------------------------
set.seed(123)
model_mlr<-train(trainSet,trainSet.y,method='multinom',trControl=fitControl,tuneLength=5)
imp = varImp(model_mlr)
imp2 <- data.frame(overall = imp$importance[,1], names= rownames(imp$importance))
confusionMatrix(model_mlr$pred$obs,model_mlr$pred$pred,positive='Y')

set.seed(123)
model_mlr.pca<-train(pca,trainSet.y,method='multinom',trControl=fitControl,tuneLength=5)
confusionMatrix(model_mlr.pca$pred$obs,model_mlr.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_mlr2<-train(trainSet[,predictors],trainSet.y,method='multinom',trControl=fitControl,tuneLength=5)
confusionMatrix(model_mlr2$pred$obs,model_mlr2$pred$pred,positive='Y')

write.csv(model_mlr$pred,'model_mlr.pred.csv')
write.csv(model_mlr.pca$pred,'model_mlr.pca.pred.csv')
write.csv(model_mlr2$pred,'model_mlr2.pred.csv')

#'
#' ## pls
## ------------------------------------------------------------------------
set.seed(123)
model_pls<-train(trainSet,trainSet.y,method='pls',trControl=fitControl,tuneLength=5)
imp = varImp(model_pls)
imp2 <- data.frame(overall = imp$importance[,1], names= rownames(imp$importance))
confusionMatrix(model_pls$pred$obs,model_pls$pred$pred,positive='Y')

set.seed(123)
model_pls.pca<-train(pca,trainSet.y,method='pls',trControl=fitControl,tuneLength=5)
confusionMatrix(model_pls.pca$pred$obs,model_pls.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$importance[,1],decreasing = T),2][1:20])
set.seed(123)
model_pls2<-train(trainSet[,predictors],trainSet.y,method='pls',trControl=fitControl,tuneLength=5)
confusionMatrix(model_pls2$pred$obs,model_pls2$pred$pred,positive='Y')

write.csv(model_pls$pred,'model_pls.pred.csv')
write.csv(model_pls.pca$pred,'model_pls.pca.pred.csv')
write.csv(model_pls2$pred,'model_pls2.pred.csv')

#'
#' ## gbm
## ------------------------------------------------------------------------
set.seed(123)
model_gbm<-train(trainSet,trainSet.y,method='gbm',trControl=fitControl,tuneLength=5)
imp = summary(model_gbm)
imp2 <- data.frame(overall = imp$rel.inf, names= rownames(imp))
confusionMatrix(model_gbm$pred$obs,model_gbm$pred$pred,positive='Y')

set.seed(123)
model_gbm.pca<-train(pca,trainSet.y,method='gbm',trControl=fitControl,tuneLength=5)
confusionMatrix(model_gbm.pca$pred$obs,model_gbm.pca$pred$pred,positive='Y')

predictors = as.vector(imp2[order(imp$rel.inf,decreasing = T),2][1:20])
set.seed(123)
model_gbm2<-train(trainSet[,predictors],trainSet.y,method='gbm',trControl=fitControl,tuneLength=5)
confusionMatrix(model_gbm2$pred$obs,model_gbm2$pred$pred,positive='Y')

write.csv(model_gbm$pred,'model_gbm.pred.csv')
write.csv(model_gbm.pca$pred,'model_gbm.pca.pred.csv')
write.csv(model_gbm2$pred,'model_gbm2.pred.csv')

#'
## ------------------------------------------------------------------------
resamps <- resamples(list(rf = model_rf, svm = model_svm, lasso = model_lasso, pls = model_pls, gbm = model_gbm, nnet = model_nnet, mlr = model_mlr, lda = model_lda))
summary(resamps)

resamps.pca <- resamples(list(rf = model_rf.pca, svm = model_svm.pca, lasso = model_lasso.pca, pls = model_pls.pca, gbm = model_gbm.pca, nnet = model_nnet.pca, mlr = model_mlr.pca, lda = model_lda.pca))
summary(resamps.pca)

resamps2 <- resamples(list(rf = model_rf2, svm = model_svm2, lasso = model_lasso2, pls = model_pls2, gbm = model_gbm2, nnet = model_nnet2, mlr = model_mlr2, lda = model_lda2))
summary(resamps2)

sink('summary-resamp-methods.txt')
summary(resamps)
sink()

sink('summary-resamp-pca-methods.txt')
summary(resamps.pca)
sink()

sink('summary-resamp-imp-methods.txt')
summary(resamps2)
sink()

#'
## ------------------------------------------------------------------------
mod = sort(resamps, decreasing = TRUE, metric = x$metric[1])[1:4]
mod = gsub('Accuracy','',mod)
mod = paste('pred_',mod,sep = '')

pred_rf<-model_rf$pred$Y[order(model_rf$pred$rowIndex)]
pred_svm<-model_svm$pred$Y[order(model_svm$pred$rowIndex)]
pred_lasso<-model_lasso$pred$Y[order(model_lasso$pred$rowIndex)]
pred_pls<-model_pls$pred$Y[order(model_pls$pred$rowIndex)]
pred_gbm<-model_gbm$pred$Y[order(model_gbm$pred$rowIndex)]
pred_nnet<-model_nnet$pred$Y[order(model_nnet$pred$rowIndex)]
pred_mlr<-model_mlr$pred$Y[order(model_mlr$pred$rowIndex)]
pred_lda<-model_lda$pred$Y[order(model_lda$pred$rowIndex)]

stack_train = cbind(pred_rf,pred_svm,pred_lasso,pred_pls,pred_gbm,pred_nnet,pred_mlr,pred_lda)
stack_train.y = rep(trainSet.y,each =3)

predictors_top<- mod
#GBM as top layer model
set.seed(123)
gbm_stacked<- train(stack_train[,predictors_top],stack_train.y,method='gbm',trControl=fitControl,tuneLength=3)
set.seed(123)
glm_stacked<- train(stack_train[,predictors_top],stack_train.y,method='glm',trControl=fitControl,tuneLength=3)
set.seed(123)
rf_stacked<- train(stack_train[,predictors_top],stack_train.y,method='rf',trControl=fitControl,tuneLength=3)

#'
#'
## ------------------------------------------------------------------------
# testSet$OOF_pred_rf<-predict(model_rf,testSet,type='prob')$Y
# testSet$OOF_pred_lasso<-predict(model_lasso,testSet,type='prob')$Y
# testSet$OOF_pred_svm<-predict(model_svm,nzv[-index,],type='prob')$Y
# testSet$OOF_pred_pls<-predict(model_pls,testSet,type='prob')$Y
# predict using GBM top layer model
# testSet$gbm_stacked<-predict(model_gbm,testSet[,predictors_top])
# confusionMatrix(testSet.y,testSet$gbm_stacked,positive='Y')

# testSet$glm_stacked<-predict(model_glm,testSet[,predictors_top])
# confusionMatrix(testSet.y,testSet$glm_stacked,positive='Y')

# testSet$rf_stacked<-predict(rf_stacked,testSet[,predictors_top])
# confusionMatrix(testSet.y,testSet$rf_stacked,positive='Y')
confusionMatrix(rf_stacked$pred$obs,rf_stacked$pred$pred,positive='Y')
confusionMatrix(glm_stacked$pred$obs,glm_stacked$pred$pred,positive='Y')
confusionMatrix(gbm_stacked$pred$obs,gbm_stacked$pred$pred,positive='Y')

pred_avg<-rowMeans(stack_train[,predictors_top])
pred_avg2<-as.factor(ifelse(pred_avg>0.5,'Y','N'))
confusionMatrix(stack_train.y,pred_avg2,positive='Y')

mod

save.image("Results-ensemble-Imput-median.RData")

write.csv(rf_stacked$pred,'rf_stacked.pred.csv')
write.csv(glm_stacked$pred$pred,'glm_stacked.pred.csv')
write.csv(gbm_stacked$pred$pred,'gbm_stacked.pred.csv')
