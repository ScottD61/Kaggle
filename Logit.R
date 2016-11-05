#Read data
train5 <- read.table("/Users/scdavis6/Documents/Work/Kaggle/Titanic/NewData/train.csv", 
                     sep = ",", na.strings = "", header = TRUE)
#Summarize data
summary(train5)
#Summary of variables
str(train5)
#Get sums of NAs per column
colSums(is.na(train5))
#Identify NA values per column

#First step - load data
#Do visualizations 
#plot histograms of each and plots of every combo



