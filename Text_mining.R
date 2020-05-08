rm(list = ls())
library(tm)
library(slam)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
libs <- c("tm","plyr","class")
lapply(libs, require, character.only = TRUE)
library(caret)
library(wordcloud)
library(rpart)
library(tree)
library(caret)
library(mlr)
library(MASS)
library(randomForest)
options(stringsAsFactors = FALSE)
#####      Student: Suhani          #########
#####  Student Number: 119220491    #########
#####       Module:CS6405            #########

###please run the code one by one##
#1.Basic Exploration
#2.Modelling classifier 1. knn 2.random Forest 3. Naive Bayes
#3. Robust Evaluation

##Note: document path is variable for location of Newsgroup folder i have used further###

documentpath="Please/Input/Newsgroup/folder/location"

##************* BASIC EXPLORATION OF DATASET    ************
Newsgroup <- c("comp.sys.ibm.pc.hardware", "sci.electronics", "talk.politics.guns", "talk.politics.misc")
newsPathNAme <- documentpath

#Build TDM
buildTDM <- function(newsType, path){
  s.dir <- sprintf("%s/%s",path,newsType)
  s.cor <- Corpus(DirSource(directory = s.dir))
  s.tdm <- TermDocumentMatrix(s.cor)
  result <- list(name=newsType, tdm = s.tdm)
}

tdm <- lapply(Newsgroup, buildTDM, path = newsPathNAme)

#Bind Newsgroup to TDM#
bindNewsgroupToTDM <- function(tdm){
  s.mat <- t(data.matrix(tdm[["tdm"]])) 
  s.df <- as.data.frame(s.mat,StringsAsFactors = FALSE)
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "newsType"
  return(s.df)
}

newsTDM <- lapply(tdm, bindNewsgroupToTDM)

#stack the label 
tdm.stack.b <- do.call(rbind.fill,newsTDM)
tdm.stack.b[is.na(tdm.stack.b)] <- 0
col_idx <- grep("newsType", names(tdm.stack.b))
tdm.stack.b <- tdm.stack.b[, c((1:ncol(tdm.stack.b))[-col_idx],col_idx)]
dim(tdm.stack.b)
#write.csv(tdm.stack.b,'/Users/suhanisingh/Desktop/Sem 2/CS6405/token_doc_basic.csv')

tdm.stack.b.df=tdm.stack.b
tdm.stack.b.df$newsType=NULL

#***************Exploration of Dataset******************************
###printing top 200 words in order of decresing frequency########
ncols = as.numeric(ncol(tdm.stack.b))
sum_of_occurences = rep(0, each=ncols-1)
for(i in 1:(ncols-1)){
  sum_of_occurences[i] = sum(as.vector(tdm.stack.b[i]))   
}
words_with_occurences = cbind.data.frame(names(tdm.stack.b.df), as.numeric(sum_of_occurences))
sorted_words_with_occurences = words_with_occurences[order(words_with_occurences[,2], decreasing = TRUE),]
names(sorted_words_with_occurences) = c("Words","Total Occurences")

for(i in 1:200){
  print(sorted_words_with_occurences[i,])
}
####Filtering token by length min. size=4, max size=20######
sorted_words_with_occurences$Words_Length = nchar(as.character(sorted_words_with_occurences[,1]))
count = 1
for(i in 1:length(sorted_words_with_occurences[,1])){
  if(sorted_words_with_occurences[i,3] >= 4 && sorted_words_with_occurences[i,3] <= 20){
    print(sorted_words_with_occurences[i,])
    count = count + 1
    if(count == 201){ 
      break
    }
  }
}
##***********Modelling different Classifiers*******************
#Selecting trainig and test set in 70:30 ratio.
train.indx <- sample(nrow(tdm.stack.b), ceiling(nrow(tdm.stack.b) * 0.7))
test.indx <- sample(1:nrow(tdm.stack.b)) [-train.indx]

##*******************Knn Modelling *****************************************************
tdm.newsType <- tdm.stack.b[,"newsType"]
tdm.stack.nl <- tdm.stack.b[, !colnames(tdm.stack.b) %in% "newsType"]
x.train.knn=tdm.stack.nl[train.indx,]
x.test.knn=tdm.stack.nl[test.indx,]
y.train.knn=tdm.newsType[train.indx]
y.test.knn=tdm.newsType[test.indx]
set.seed(6405)
knn.model.b.knn<- knn(x.train.knn,x.test.knn,y.train.knn)
conf.mat.b.knn <- table(knn.model.b,y.test.knn )
conf.mat.b.knn
(acuracy.knn.b <- sum(diag(conf.mat.b)) / length(test.indx) * 100) #80.00
acuracy.knn.b
###computing Precision and Recall####
n = sum(conf.mat.b.knn) # number of instances
nc = nrow(conf.mat.b.knn) # number of classes
diag = diag(conf.mat.b.knn) # number of correctly classified instances per class 
rowsums = apply(conf.mat.b.knn, 1, sum) # number of instances per class
colsums = apply(conf.mat.b.knn, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
accuracy.knn = sum(diag) / n 
accuracy.knn
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 

###************Random Forest Modelling*****************************************************
train <- tdm.stack.b[train.indx, ]
test <- tdm.stack.b[test.indx, ]
train$newsType <- as.factor(train$newsType)
test$newsType=as.factor(test$newsType)

set.seed(6405)
classifier.rf<- randomForest(x = train,y = train$newsType,nTree = 500)
rf.pred <- predict(classifier.rf,test)
conf.mat.rf <- table(test[,"newsType"], rf.pred)
conf.mat.rf
acuracy.rf <- sum(diag(conf.mat.rf)) / length(test.indx) * 100
acuracy.rf
###computing Precision and Recall####
n = sum(conf.mat.rf) # number of instances
nc = nrow(conf.mat.rf) # number of classes
diag = diag(conf.mat.rf) # number of correctly classified instances per class 
rowsums = apply(conf.mat.rf, 1, sum) # number of instances per class
colsums = apply(conf.mat.rf, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 

#***********************************Naive Bayes Classifier*********************************************
newstype <- c("comp.sys.ibm.pc.hardware", "sci.electronics", "talk.politics.guns", "talk.politics.misc")
newsPathNAme <- documentpath
training_folder <- documentpath
##Modifying data as required for naive Bayes#
folder_mining <- function(subfolder) {
  tibble(file = dir(subfolder, full.names = TRUE)) %>%
    mutate(full_text = map(file, read_file)) %>%        #Generate new element
    transmute(id = basename(file), full_text) %>%       #Add new columns by dropping existing ones:
    unnest(full_text)
}


raw_text_corpus <- tibble(folder = dir(training_folder, full.names = TRUE)) %>%
  mutate(folder_out = map(folder, folder_mining)) %>%
  unnest(cols = c(folder_out)) %>%
  transmute(full_text =full_text,Newsgroup_Class = basename(folder))

hardware_freq <- subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "comp.sys.ibm.pc.hardware")
elect_freq <- subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "sci.electronics")
guns_freq <- subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "talk.politics.guns")
misc_freq <- subset(raw_text_corpus,raw_text_corpus$Newsgroup_Class == "talk.politics.misc")


set.seed(6405)
for(k in 1:100){
  sample_h <- sample.int(n=100, size=70)
  sample_s <- sample.int(n=100, size=70)
  sample_g <- sample.int(n=100, size=70)
  sample_m <- sample.int(n=100, size=70)}
train.set <- rbind(hardware_freq[sample_h,], elect_freq[sample_s,], guns_freq[sample_g,], misc_freq[sample_m,])
test.set  <- rbind(hardware_freq[-sample_h,], elect_freq[-sample_s,], guns_freq[-sample_g,], misc_freq[-sample_m,])
actual_test_class <- test.set$Newsgroup_Class
test_text <- test.set$full_text
actual_train_class <- train.set$Newsgroup_Class
train_text <- train.set$full_text

buildCorpus <- function(Newsgroup_Class, train.subset){
  tp = toString(Newsgroup_Class)
  train.new <- subset(train.subset,train.subset$Newsgroup_Class == tp)
  s.cor <- Corpus(VectorSource(train.new$full_text))
  s.tdm <- TermDocumentMatrix(s.cor)
  result <- list(name=Newsgroup_Class, tdm = s.tdm)
}

bindnewstypeToTDM <- function(tdm){
  s.mat <- t(data.matrix(tdm[["tdm"]])) 
  s.df <- as.data.frame(s.mat,StringsAsFactors = FALSE)
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "Newsgroup_Class"
  return(s.df)
}
train.tdm <- lapply(newstype, buildCorpus, train.set)
train_cand_TDM <- lapply(train.tdm, bindnewstypeToTDM)
train.stack <- do.call(rbind.fill,train_cand_TDM)
train.stack[is.na(train.stack)] <- 0
col_idx <- grep("Newsgroup_Class", names(train.stack))
train.stack <- train.stack[, c((1:ncol(train.stack))[-col_idx],col_idx)]

test.tdm <- lapply(newstype, buildCorpus, test.set)
test_cand_TDM <- lapply(test.tdm, bindnewstypeToTDM)
test.stack <- do.call(rbind.fill,test_cand_TDM)
test.stack[is.na(test.stack)] <- 0
col_idx <- grep("Newsgroup_Class", names(test.stack))
test.stack <- test.stack[, c((1:ncol(test.stack))[-col_idx],col_idx)]
#write.csv(test.stack,'/Users/suhanisingh/Desktop/Sem 2/CS6405/test.stack.csv')

hardware_row_prob_ZeroEx<-as.data.frame(test.stack)
electronics_row_prob_ZeroEx<-as.data.frame(test.stack)
guns_row_prob_ZeroEx<-as.data.frame(test.stack)
misc_row_prob_ZeroEx<-as.data.frame(test.stack)
total_hardware_word=0;total_misc_word=0
total_vocabulary=0
total_electronics_word=0;total_guns_word=0

# Function Trainning Naive Base
train_naive_bayes <- function(tdm.stack) {
  tdm.stack=train.stack
  comp_hardware_freq <- subset(tdm.stack,tdm.stack$Newsgroup_Class == "comp.sys.ibm.pc.hardware")
  comp_hardware_freq <- as.data.frame(comp_hardware_freq)
  col_idx <- grep("Newsgroup_Class", names(comp_hardware_freq))
  hardware_row_sums <- comp_hardware_freq[, c((1:ncol(comp_hardware_freq))[-col_idx])]
  demo = data.frame(hardware_row_sums[0,])
  hardware_row_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(hardware_row_sums, na.rm = T))
  head(hardware_row_sums)
  #r_sum W/O Zero valued row
  hardware_row_sums <- as.data.frame(hardware_row_sums)
  #hardware_row_sums <- hardware_row_sums[,-1] 
  hardware_row_sums_ZeroEx <- hardware_row_sums[hardware_row_sums$frequency != 0, ]
  total_hardware_word <- sum(hardware_row_sums$frequency)
  summary(hardware_row_sums)
  head(hardware_row_sums_ZeroEx)
  
  # Calculating Electronics row sum
  electronics_freq <- subset(tdm.stack,tdm.stack$Newsgroup_Class == "sci.electronics")
  electronics_freq <- as.data.frame(electronics_freq)
  col_idx <- grep("Newsgroup_Class", names(electronics_freq))
  electronics_row_sums <- electronics_freq[, c((1:ncol(electronics_freq))[-col_idx])]
  demo = data.frame(electronics_row_sums[0,])
  electronics_row_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(electronics_row_sums, na.rm = T))
  #r_sum W/O Zero valued row
  electronics_row_sums <- as.data.frame(electronics_row_sums)
  electronics_row_sums_ZeroEx <- electronics_row_sums[electronics_row_sums$frequency != 0, ]
  total_electronics_word <- sum(electronics_row_sums$frequency)
  head(electronics_row_sums)
  
  # Calculating Guns row sum
  politics_guns_freq <- subset(tdm.stack,tdm.stack$Newsgroup_Class == "talk.politics.guns")
  col_idx <- grep("Newsgroup_Class", names(politics_guns_freq))
  guns_row_sums <- politics_guns_freq[, c((1:ncol(politics_guns_freq))[-col_idx])]
  demo = data.frame(guns_row_sums[0,])
  guns_row_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(guns_row_sums, na.rm = T))
  #r_sum W/O Zero valued row
  guns_row_sums <- as.data.frame(guns_row_sums)
  guns_row_sums_ZeroEx <- guns_row_sums[guns_row_sums$frequency != 0, ]
  total_guns_word <- sum(guns_row_sums$frequency)
  head(guns_row_sums)
  
  # Calculating Misc row sum
  politics_misc_freq <- subset(tdm.stack,tdm.stack$Newsgroup_Class == "talk.politics.misc")
  col_idx <- grep("Newsgroup_Class", names(politics_misc_freq))
  misc_row_sums <- politics_misc_freq[, c((1:ncol(politics_misc_freq))[-col_idx])]
  demo = data.frame(misc_row_sums[0,])
  misc_row_sums <- data.frame(colname = names(demo),frequency = slam::col_sums(misc_row_sums, na.rm = T))
  #r_sum W/O Zero valued row
  misc_row_sums <- as.data.frame(misc_row_sums)
  misc_row_sums_ZeroEx <- misc_row_sums[misc_row_sums$frequency != 0, ]
  total_misc_word <- sum(misc_row_sums$frequency)
  head(misc_row_sums)
  
  # Calculating Misc row sum
  total_row_sums <- tdm.stack[, c((1:ncol(tdm.stack))[-col_idx])]
  total_row_sums <- slam::col_sums(total_row_sums, na.rm = T)
  total_row_sums <- as.data.frame(total_row_sums)
  total_vocabulary <- nrow(total_row_sums)
  
  #Hardware Probability Calculations
  hardware_row_sums_ZeroEx <- as.data.frame(hardware_row_sums_ZeroEx)
  hardware_probs <- (hardware_row_sums_ZeroEx$frequency+1)/(total_hardware_word+total_vocabulary) 
  hardware_row_prob_ZeroEx <- cbind(hardware_row_sums_ZeroEx,hardware_probs)
  
  #Electronics Probability Calculations
  electronics_row_sums_ZeroEx <- as.data.frame(electronics_row_sums_ZeroEx)
  electronics_probs <- (electronics_row_sums_ZeroEx$frequency+1)/(total_electronics_word+total_vocabulary) 
  electronics_row_prob_ZeroEx <- cbind(electronics_row_sums_ZeroEx,electronics_probs)
  
  #Guns Probability Calculations
  guns_row_sums_ZeroEx <- as.data.frame(guns_row_sums_ZeroEx)
  guns_probs <- (guns_row_sums_ZeroEx$frequency+1)/(total_guns_word+total_vocabulary) 
  guns_row_prob_ZeroEx <- cbind(guns_row_sums_ZeroEx,guns_probs)
  
  #Misc Probability Calculations
  misc_row_sums_ZeroEx <- as.data.frame(misc_row_sums_ZeroEx)
  misc_probs <- (misc_row_sums_ZeroEx$frequency+1)/(total_misc_word+total_vocabulary) 
  misc_row_prob_ZeroEx <- cbind(misc_row_sums_ZeroEx,misc_probs)
  
}

Naive_Bayes_classifier <- function(string) {
  list = unlist(strsplit(string," "))
  hardware_prob = 0
  electronics_prob = 0
  guns_prob = 0
  misc_prob = 0
  
  #hardware_word_probs <- format(read.csv("/Users/suhanisingh/Desktop/Sem 2/CS6405/hardware_probs.csv",header=T,sep=","), scientific = FALSE)
  #electronics_word_probs <- format(read.csv("/Users/suhanisingh/Desktop/Sem 2/CS6405/Newsgroups/electronics_probs.csv",header=T,sep=","), scientific = FALSE)
  #guns_word_probs <- format(read.csv("/Users/suhanisingh/Desktop/Sem 2/CS6405/guns_probs.csv",header=T,sep=","), scientific = FALSE)
  #misc_word_probs <-  format(read.csv("/Users/suhanisingh/Desktop/Sem 2/CS6405/misc_probs.csv",header=T,sep=","), scientific = FALSE)
  hardware_word_probs <- format(hardware_row_prob_ZeroEx, scientific = FALSE)
  format(hardware_word_probs, scientific = FALSE)
  electronics_word_probs <- electronics_row_prob_ZeroEx
  guns_word_probs <- guns_row_prob_ZeroEx
  misc_word_probs <-  misc_row_prob_ZeroEx
  
  list = as.list(strsplit(string, '\\s+')[[1]])
  
  
  for (word in list) {
    if(length(hardware_word_probs[hardware_word_probs$colname== word,]$hardware_probs) != 0 && word %in% hardware_word_probs$colname){
      hardware_prob = hardware_prob + log(as.numeric(hardware_word_probs[hardware_word_probs$colname== word,]$hardware_probs))
    }else{
      hardware_prob = hardware_prob + log((1/(total_hardware_word+total_vocabulary)))
    }
    if(length(electronics_word_probs[electronics_word_probs$colname== word,]$electronics_probs) != 0 && word %in% electronics_word_probs$colname){
      electronics_prob = electronics_prob + log(as.numeric(electronics_word_probs[electronics_word_probs$colname== word,]$electronics_probs))
    }else{
      electronics_prob = electronics_prob + log((1/(total_electronics_word+total_vocabulary))) 
    }
    if(length(guns_word_probs[guns_word_probs$colname== word,]$guns_probs) != 0 && word %in% guns_word_probs$colname){
      guns_prob = guns_prob + log(as.numeric(guns_word_probs[guns_word_probs$colname== word,]$guns_probs))
    }else{
      guns_prob = guns_prob + log((1/(total_guns_word+total_vocabulary)))
    } 
    if(length(misc_word_probs[misc_word_probs$colname== word,]$misc_probs) != 0 && word %in% misc_word_probs$colname){
      misc_prob = misc_prob + log(as.numeric(misc_word_probs[misc_word_probs$colname== word,]$misc_probs))
    }else{
      misc_prob = misc_prob + log((1/(total_misc_word+total_vocabulary)))
    }
  }
  
  results <- c(hardware_prob,electronics_prob,guns_prob,misc_prob)
  largest <- max(results)
  
  if (largest== hardware_prob){
    classification <- "comp.sys.ibm.pc.hardware"
  } else if(largest== electronics_prob) {
    classification <- "sci.electronics"
  } else if (largest== guns_prob) {
    classification <- "talk.politics.guns"
  } else {
    classification <- "talk.politics.misc"
  } 
  
  return(classification)
}

train_naive_bayes(train.stack)
NB_test_predictions <- unlist(lapply(test_text,Naive_Bayes_classifier))
NB_Confusion <-table(actual_test_class,NB_test_predictions)
NB_Confusion
#computing accuracy
NB_accuracy <- sum(diag(NB_Confusion))/sum(NB_Confusion)
NB_accuracy ##0.64166

###computing Precision and Recall####
n = sum(NB_Confusion) # number of instances
nc = nrow(NB_Confusion) # number of classes
diag = diag(NB_Confusion) # number of correctly classified instances per class 
rowsums = apply(NB_Confusion, 1, sum) # number of instances per class
colsums = apply(NB_Confusion, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1) 

#***************Robust Evaluation*******************************************
rm(list = ls())
Newsgroup <- c("comp.sys.ibm.pc.hardware", "sci.electronics", "talk.politics.guns", "talk.politics.misc")
newsPathNAme <- documentpath

# "Function for CLEANING CORPUS"
cleanCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp,removeNumbers)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  #corpus.tmp <- tm_map(corpus.tmp, function(x) iconv(x, "latin1", "ASCII", sub=""))
  return(corpus.tmp)
}

# "BUILD TDM"
buildTDM <- function(newsType, path){
  s.dir <- sprintf("%s/%s",path,newsType)
  s.cor <- Corpus(DirSource(directory = s.dir))
  s.cor.cl <- cleanCorpus(s.cor)
  s.tdm <- TermDocumentMatrix(s.cor.cl)
  #s.tdm <- removeSparseTerms(s.tdm, 0.7)
  result <- list(name=newsType, tdm = s.tdm)
}

tdm <- lapply(Newsgroup, buildTDM, path = newsPathNAme)

#"Bind Newsgroup to TDM"
bindNewsgroupToTDM <- function(tdm){
  s.mat <- t(data.matrix(tdm[["tdm"]])) 
  s.df <- as.data.frame(s.mat,StringsAsFactors = FALSE)
  
  s.df <- cbind(s.df, rep(tdm[["name"]], nrow(s.df)))
  colnames(s.df)[ncol(s.df)] <- "newsType"
  return(s.df)
}

newsTDM <- lapply(tdm, bindNewsgroupToTDM)

#stack classifying labels
tdm.stack.R <- do.call(rbind.fill,newsTDM)
tdm.stack.R[is.na(tdm.stack.R)] <- 0
col_idx <- grep("newsType", names(tdm.stack.R))
tdm.stack.R <- tdm.stack.R[, c((1:ncol(tdm.stack.R))[-col_idx],col_idx)]

#write.csv(tdm.stack.R,'/Users/suhanisingh/Desktop/Sem 2/CS6405/token_doc_robust.csv')
#Another variable
tdm.stack.R.df=tdm.stack.R
tdm.stack.R.df$newsType=NULL

# Training and test set For Robust evaluation
train.indx.R <- sample(nrow(tdm.stack.R), ceiling(nrow(tdm.stack.R) * 0.7))
test.indx.R <- sample(1:nrow(tdm.stack.R)) [-train.indx.R]

train <- tdm.stack.R[train.indx.R, ]
test <- tdm.stack.R[test.indx.R, ]
train$newsType <- as.factor(train$newsType)
test$newsType=as.factor(test$newsType)
## Important Feature Selection Using Recursive Feature Elimination(rfe)
set.seed(6405)
subsets<- c(1000,2000, ncol(tdm.stack.R))
ctrl <- rfeControl(functions = rfFuncs, method = "cv",number = 10,verbose = FALSE)
length(tdm.stack) #19615
rf.rfe <- rfe(train, train$newsType,sizes = subsets,rfeControl = ctrl)
length(rf.rfe$optVariables)
imp_var=rf.rfe$optVariables
imp_var
#***********Making New dataset with selected features******************
set.seed(6405)
newtrain=train[,imp_var]
newtest=test[,imp_var]

##Calculating best value of k for knn Model
par(mfrow=c(1,1))
Kmax = 40
 acc = numeric(Kmax)
for(k in 1:Kmax){
  ko = knn(tdm.stack.nl[train.indx.R,], tdm.stack.nl[test.indx.R,], tdm.newsType[train.indx.R], k)
  tb = table(ko,tdm.newsType[test.indx.R])
  acc[k] = sum(diag(tb)) / sum(tb)	
}

#****Hyperparameter tuning for knn with k=1 and Features selected above***************
set.seed(6405)
newtrain1=newtrain
newtest1=newtest
newtrain1$newsType=NULL
newtest1$newsType=NULL
knn.model.rfe<- knn(newtrain1,newtest1,newtrain$newsType,k=1)
conf.mat.knn.rfe <- table(knn.model.rfe,newtest$newsType )
conf.mat.knn.rfe
(acuracy.knn.rfe <- sum(diag(conf.mat.knn.rfe)) / length(test.indx.R) * 100) #85%
###computing Precision and Recall####
n = sum(conf.mat.knn.rfe) # number of instances
nc = nrow(conf.mat.knn.rfe) # number of classes
diag = diag(conf.mat.knn.rfe) # number of correctly classified instances per class 
rowsums = apply(conf.mat.knn.rfe, 1, sum) # number of instances per class
colsums = apply(conf.mat.knn.rfe, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)

##****Hyperparameter tuning for Random forest and Features selected above***************
set.seed(6405)
classifier.rfe <- randomForest(x = newtrain, y = newtrain$newsType,nTree = 5000)
y_pred.rfe <- predict(classifier.rfe,newtest)
cm.rf.rfe <- table(newtest[,"newsType"], y_pred.rfe)
cm.rf.rfe
acuracy.rf.hy <- (sum(diag(cm.rf.rfe)) / length(test.indx)) * 100
acuracy.rf.hy ##99.16667
###computing Precision and Recall####
n = sum(cm.rf.rfe) # number of instances
nc = nrow(cm.rf.rfe) # number of classes
diag = diag(cm.rf.rfe) # number of correctly classified instances per class 
rowsums = apply(cm.rf.rfe, 1, sum) # number of instances per class
colsums = apply(cm.rf.rfe, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)


##*******Decison tree fit and training ******************
set.seed(6405)
#Decision tree
library(rpart)
attach(newtrain)
model_tree = rpart(newsType~.,data =newtrain,method = "class")
pred_tree = predict(model_tree,newtest1,type = "class")
cf.dec.tree=table(newtest$newsType,pred_tree)
(acuracy.tree <- sum(diag(cf.dec.tree)) / length(test.indx.R) * 100) #98.33
printcp(model_tree)

#create a task
trainTask <- makeClassifTask(data = newtrain,target = "newsType")
testTask <- makeClassifTask(data = newtest, target = "newsType")

#make tree learner
makeatree <- makeLearner("classif.rpart", predict.type = "response")
##set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 10L)
dim(newtrain)

#Search for hyperparameters
gs <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)

#do a grid search
gscontrol <- makeTuneControlGrid()

#hypertune the parameters
stune <- tuneParams(learner = makeatree, resampling = set_cv, task = trainTask, par.set = gs, control = gscontrol, measures = bac)
#Result: minsplit=37; minbucket=45; cp=0.156 : bac.test.mean=0.9864773
# stune$x
# $minsplit
# [1] 37
# 
# $minbucket
# [1] 45
# 
# $cp
# [1] 0.1557778
# stune$y
# bac.test.mean 
# 0.9864773
#using hyperparameters for modeling
t.tree <- setHyperPars(makeatree, par.vals = stune$x)
#train the model
t.rpart <- train(t.tree, trainTask)
getLearnerModel(t.rpart)
#make predictions
tpmodel <- predict(t.rpart, testTask)
tpmodel$data$response
cf.dec.tree=table(newtest$newsType,tpmodel$data$response)
acc.dec.tree.hy=(sum(diag(cf.dec.tree)) / length(test.indx.R) * 100)
acc.dec.tree.hy
cf.dec.tree
n = sum(cf.dec.tree) # number of instances
nc = nrow(cf.dec.tree) # number of classes
diag = diag(cf.dec.tree) # number of correctly classified instances per class 
rowsums = apply(cf.dec.tree, 1, sum) # number of instances per class
colsums = apply(cf.dec.tree, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)


##*********Support Vector Machine for Robust Model******************
set.seed(6405)
getParamSet("classif.ksvm") #do install kernlab package 
ksvm <- makeLearner("classif.ksvm", predict.type = "response")

##Set parameters
pssvm <- makeParamSet(
  makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
  makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)

##specify search function
ctrl <- makeTuneControlGrid()
res <- tuneParams(ksvm, task = trainTask, resampling = set_cv, par.set = pssvm, control = ctrl,measures = bac)
#C=1; sigma=0.00390625 : bac.test.mean=0.6718443
 #res$y
# bac.test.mean 
# 0.6718443
 res$x
# $C
# [1] 1
# 
# $sigma
# [1] 0.00390625
#set the model with best params
t.svm <- setHyperPars(ksvm, par.vals = res$x)
#train
par.svm <- train(ksvm, trainTask)
#test
predict.svm <- predict(par.svm, testTask)
cf.svm.hy=table(newtest$newsType,predict.svm$data$response)
cf.svm.hy
acc.svm.hy=(sum(diag(cf.svm.hy)) / length(test.indx.R) * 100)
acc.svm.hy #95
###computing Precision and Recall####
n = sum(cf.svm.hy) # number of instances
nc = nrow(cf.svm.hy) # number of classes
diag = diag(cf.svm.hy) # number of correctly classified instances per class 
rowsums = apply(cf.svm.hy, 1, sum) # number of instances per class
colsums = apply(cf.svm.hy, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes
precision = diag / colsums 
recall = diag / rowsums  
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)


