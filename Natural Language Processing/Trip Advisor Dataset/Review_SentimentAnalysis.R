install.packages("NLP")
install.packages("tm")
library(NLP)
library(tm)
install.packages("SnowballC")
library(SnowballC)
install.packages("csv")
library(csv)
write.xlsx(mydata, "c:/mydata.xlsx")
library(rpart)
library(rpart.plot)
library(caTools)

BRP_Sample = read.csv("Text review_R.csv", stringsAsFactors=FALSE)

corpus_Sample = Corpus(VectorSource(BRP_Sample$Customer.Review))

corpus_Sample = tm_map(corpus_Sample, tolower)
strwrap(corpus_Sample[[1]])

corpus_Sample = tm_map(corpus_Sample, PlainTextDocument)

corpus_Sample = tm_map(corpus_Sample, removePunctuation)

# Look at stop words 
#stopwords("english")[1:15]

# Remove stopwords and repetitive words
#corpus_Sample = tm_map(corpus_Sample, removeWords, c("hotel","restaurant","stayed","travelled", stopwords("english")))
corpus_Sample = tm_map(corpus_Sample, removeWords, c("2017","2015","2016","poor","never","great","like", "even", "bad", "good", stopwords("english")))
#corpus_Sample2 = tm_map(corpus_Sample1, removeWords, c("poor","never","great","like"))

#strwrap(corpus_Sample[[1]])


#Creates a matrix with 24(Number of customer reviews) entries against each word 
frequencies = DocumentTermMatrix(corpus_Sample)
inspect(frequencies[110:115, 1:10])

#Puts boundary to lowest frequeny of stemmed word
findFreqTerms(frequencies, lowfreq=20)
inspect(frequencies[20:24, 1:10])

sparse_Sample = removeSparseTerms(frequencies, 0.995) #0.9965
ocrSparse_Sample = as.data.frame(as.matrix(frequencies))

# Now let's convert the sparse matrix into a data frame that we'll be able to use for our predictive models.


# Make all variable names R-friendly

colnames(ocrSparse_Sample) = make.names(colnames(ocrSparse_Sample))

ocrSparse_Sample$Sentiment = BRP_Sample$Sentiment

set.seed(110)

split_Sample = sample.split(ocrSparse_Sample$Sentiment, SplitRatio = 0.9)

trainSparse_Sample = subset(ocrSparse_Sample, split_Sample==TRUE)
testSparse_Sample = subset(ocrSparse_Sample, split_Sample==FALSE)


tweetCART_Sample = rpart(Sentiment ~ ., data=trainSparse_Sample, method="class")
prp(tweetCART_Sample)


# Evaluate the performance of the model
predictCART = predict(tweetCART_Sample, newdata=testSparse_Sample, type="class")
summary(predictCART)
View(testSparse_Sample)
table(testSparse_Sample$Sentiment, predictCART)


