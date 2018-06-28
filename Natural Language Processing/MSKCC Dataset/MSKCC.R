install.packages("NLP")
install.packages("tmap")
install.packages("tm")
install.packages("SnowballC")
install.packages("csv")
#library(csv)
#write.xlsx(mydata, "c:/mydata.xlsx")
install.packages("rpart")
#library(rpart.plot)
install.packages("caTools")
install.packages("nnet")
install.packages("mice")



library(csv)
library(NLP)
library(tmap)
library(tm)
library(SnowballC)
library(rpart)
library(caTools)
library(nnet)
library(mice)
  
# installing/loading the package:
if(!require(installr)) {
  install.packages("installr"); require(installr)} #load / install+load installr

# using the package:





#BRP_Sample = read.csv("R_DataFrame_OCR.csv", stringsAsFactors=FALSE)
training_text = read.csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training.csv", stringsAsFactors=FALSE)
test_text = read.csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test.csv", stringsAsFactors=FALSE)
consolidated = read.csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/consolidated.csv", stringsAsFactors=FALSE)
summary(consolidated)

consolidated$Class[1]

#drop(training_text)
#str(training_text)
#str(test_text)

#keeps_train = c("Text", "Class")
#training_text[keeps_train]

#keeps_test = c("Text")
#text_text[keeps_test]


# str(training_text)
# summary(training)
# str(test)
# summary(test)


corpus_Sample = Corpus(VectorSource(consolidated$Text))

#Converting the entire text into lower cases
corpus_Sample = tm_map(corpus_Sample, tolower)
str(corpus_Sample)

strwrap(corpus_Sample[[8989]])


#Removing Whitespaces
#corpus_Sample = tm_map(corpus_Sample, PlainTextDocument)

#Removing punctutations
corpus_Sample = tm_map(corpus_Sample, removePunctuation)

#Remove stop words
corpus_Sample = tm_map(corpus_Sample, removeWords, c(stopwords("english")))



TermMatrix = DocumentTermMatrix(corpus_Sample)
inspect(TermMatrix[110:115, 1:10])

#Puts boundary to lowest frequeny of stemmed word
findFreqTerms(TermMatrix, lowfreq=20)
inspect(TermMatrix[20:24, 1:10])


#Below resulting matrix contains only terms with a sparse factor of less than sparse
sparseTermMatrix = removeSparseTerms(TermMatrix, 0.9) #0.9965

# Now let's convert the sparse matrix into a data frame that we'll be able to use for our predictive models.
MSKCC_TextToTermDF = as.data.frame(as.matrix(sparseTermMatrix))
#View(MSKCC_TrainingDF)







# Make all variable names R-friendly

colnames(MSKCC_TextToTermDF) = make.names(colnames(MSKCC_TextToTermDF))

str(MSKCC_TextToTermDF)

#MSKCC_TrainingDF$Gene = training$Gene
#MSKCC_TrainingDF$Variation = training$Variation
#MSKCC_TextToTermDF$Class = MSKCC_TextToTermDF$Class

#ocrSparse_Sample$Sentiment = BRP_Sample$Sentiment

write.csv(MSKCC_TextToTermDF, file = "C:/Users/LS/Downloads/Kaggle/MSKCC data/MSKCC_TextToTermDF.csv")


set.seed(110)
N = 3321/8989

split_Train = MSKCC_TextToTermDF[1:3321,]
split_Test = MSKCC_TextToTermDF[3322:8989,]
summary(split_Train)

write.csv(split_Train, file = "C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Train.csv")
write.csv(split_Test, file = "C:/Users/LS/Downloads/Kaggle/MSKCC data/split_Test.csv")




split_Train$Class = consolidated$Class[1:3321]
summary(consolidated)
consolidated$Class[3322]
# split_Sample = sample.split(MSKCC_TextToTermDF$Class, SplitRatio = N, group = NULL)
# 
# MSKCC_TrainingDF_Train = subset(MSKCC_TextToTermDF, split_Sample==TRUE)
# MSKCC_TrainingDF_Test = subset(MSKCC_TextToTermDF, split_Sample==FALSE)

# mice(data = split_Train, MaxNWts = 30900)
# 
# multifit = multinom(Class ~ .,data = split_Train)
# 
# model <- nnet(Class~.,data = split_Train,family="multinomial",size = 30900,softmax=TRUE,MaxNWts =1)

#tweetCART_Sample = rpart(Class ~ ., data=trainSparse_Sample, method="class")
#prp(tweetCART_Sample)


# Evaluate the performance of the model 
predictCART = predict(multifit, newdata=MSKCC_TrainingDF_Test, type="class")
summary(predictCART)
View(MSKCC_TrainingDF_Test)
table(MSKCC_TrainingDF_Test$CLass,predictCART)

#table(testSparse_Sample$Sentiment, predictCART)
(20+18)/(20+18+7+5)


#########################################################################
#########################################################################
#########################################################################
#########################################################################



training = read.csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/training.csv", stringsAsFactors=FALSE)
test = read.csv("C:/Users/LS/Downloads/Kaggle/MSKCC data/test.csv", stringsAsFactors=FALSE)

str(training)
summary(training)
str(test)
summary(test)







