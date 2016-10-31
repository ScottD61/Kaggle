#Q2
x <- c(0.8, 0.47, 0.51, 0.73, 0.36, 0.58, 0.57, 0.85, 0.44, 0.42)
y <- c(1.39, 0.72, 1.55, 0.48, 1.19, -1.59, 1.23, -0.65, 1.49, 0.05)

mydata <- data.frame(x,y)
n <- nrow(mydata)
xy <- x*y
m <- (n*sum(xy)-sum(x)*sum(y)) / (n*sum(x^2)-sum(x)^2)
m

a <- lm(I(x - mean(x))~ I(y -mean(y)) -1, data = mydata)
a

#Q3
#reproducible example
wt <- c(4, 6, 12)
mpg <- c(20, 10, 8)

mydata2 <- data.frame(x,y)

car1 <- lm(mpg ~ wt, mydata2)
car1

#actual answer
car <- lm(mpg ~ wt, mtcars)
summary(car)

#Q6
x <- c(8.58, 10.46, 9.01, 9.64, 8.86)

mean(xtransformed) = 0
var(xtransformed) = 1

xtransformed<-(x-mean(x))/sd(x)



#Q7
x1 <- c(0.8, 0.47, 0.51, 0.73, 0.36, 0.58, 0.57, 0.85, 0.44, 0.42)
y1 <- c(1.39, 0.72, 1.55, 0.48, 1.19, -1.59, 1.23, -0.65, 1.49, 0.05)

mydata1 <- data.frame(x1, y1)

model <- lm(y ~ x, mydata1)
model