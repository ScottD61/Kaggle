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

featureEngineer <- function(bikes)    {
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

#To do - take out dummy coding b/c lm() function does this automatically
library(dummies)
#Convert variables with factors to binary in training set
Bin <- dummy.data.frame(train) 
#Convert variables with factors to binary in test set
Bint <- dummy.data.frame(test) 
#Drop timestamp and casual variables from test set
Bin1 <- subset(Bin, select = c(2:19, 21:53))

#Linear regression of casual 
model1 <- lm(casual ~ ., data = Bin1)
#Get summary of model1
summary(model1)
#Get confidence intervals for model 1
confint(model1, conf.level = 0.95)
#Plot regression


#Predict value of casual
#Create vector of zero values in test set for registered variable
Bint$registered <- c(0)
#Create count column in test set 
Bint$casual <- predict(model1, Bint)


#Linear regression of registered
model2 <- lm(registered ~ ., data = Bin1)
#Get summary of model1
summary(model2)
#Get confidence intervals for model 1
confint(model2, conf.level = 0.95)
#Plot regression
plot(model2)

#Create casual column in test set
#Non-binarized predictive model for casual rentals
train1 <- subset(train, select = c(2:11, 13:15))

model1.5 <- lm(casual ~ ., data = train1)
summary(model1.5)
test$registered <- c(0)
#Create count column in test set 
test$casual <- predict(model1.5, test)

#Create .csv of predictions
write.csv(test, file = "/Users/scdavis6/Documents/Work/Kaggle/Bikes/results.csv")

