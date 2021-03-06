#Load dataset
#First 1k rows
#Training set
bikes <- read.table("/Users/scdavis6/Documents/Work/Kaggle/Bikes/train.csv", sep = ",",
                    header = TRUE, na.strings = "NA")
#Test set
bikes2 <- read.table("/Users/scdavis6/Documents/Work/Kaggle/Bikes/test.csv", sep = ",",
                     header = TRUE, na.strings = "NA")
#Descriptive statistics
#Get data type
class(bikes)
#Get variable names
ls(bikes)
#Get data types
str(bikes)
#Summary stats
summary(bikes)
#Create time series object
#bikes.ts <- ts
#Descriptive plots
plot(bikes$count, ylab = "count")
#Scatterplot Matrix
pairs(~count + temp + humidity + windspeed, data = bikes,
      main = "Count Scatterplot Matrix")

#Create function for preparting data 

featureEngineer <- function(bikes)	{
#Convert certain variables to factors
names <- c("season", "holiday", "workingday", "weather")
bikes[,names] <- lapply(bikes[,names],	factor)
#Extract three types of variables from datetime variable
#Create weekday variable from date.time
#Convert datetime to datetype POXIXIt
bikes$datetime <- as.character(bikes$datetime)
bikes$datetime <- strptime(bikes$datetime,	format = "%Y-%m-%d	%T", tz = "EST")
#Convert datetime to hours
bikes$hour <- as.integer(substr(bikes$datetime,	12,13))
bikes$hour <- as.factor(bikes$hour)
#Get weekday for each date from datetime - REPORT ERROR
bikes$weekday <- as.factor(weekdays(bikes$datetime))
bikes$weekday <- factor(bikes$weekday,	levels	=	c("Monday",	"Tuesday",	"Wednesday",	
                                            "Thursday",	"Friday",	"Saturday",	"Sunday"))
#Get year from data
bikes$year <- as.integer(substr(bikes$datetime,	1,4))
bikes$year <- as.factor(bikes$year)

    return(bikes)
}

train <- featureEngineer(bikes)
test <- featureEngineer(bikes2)

#to do - get random forest ideas
#graphics create season-x, y- multiple variables(casual, registered, count), z - 


#For random forest
library(randomForest)
#Time system
begtime <- Sys.time()
#set.seed
set.seed(304)
#Create variables as parameters for random forest
myNtree = 500 
myMtry = 5
myImportance = TRUE
#Remove registered and count variables from train dataset
testTrain <- subset(train, select = -c(datetime, count, registered))
#Create random forest using randomForest function
#Predicts casual registrations
ranbikes <- randomForest(casual ~., data = testTrain, ntree = myNtree, 
                         mtry = myMtry, importance = myImportance)
#Get runtime
runTime <- Sys.time() -begtime
#Get time
runTime
#Get importance of each variable for randomForest - redo operation
varImpPlot(ranbikes, main = "Fit of Casual")

#Randomforest for registered user predictions

             


