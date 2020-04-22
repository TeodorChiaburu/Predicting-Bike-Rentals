# Analyzing a bike sharing dataset with time series
# Teodor Chiaburu, 900526



#------------------------ Data preprocessing ------------------------#

# Import dataset
bikes <- read.csv("BikeSharing.csv")
bikes <- bikes[-1] # first feature is redundant
bikes <- bikes[-c(13, 14)] # keep only total rentals as output
                           # discard casual and registered rentals

# Plot total rentals 
plot(bikes$dteday, bikes$cnt, main = "Bike rentals 2011-2012", 
     xlab = "Date", ylab = "Total rentals")
boxplot(bikes$cnt~bikes$season, main = "Bike rentals per season",
        xlab = "Season", ylab = "Total rentals")
plot(bikes$temp, bikes$cnt, main="Bike rentals 2011-2012", 
     xlab = "Temperature", ylab = "Total rentals")
rentalsWeather <- table(bikes$weathersit)
barplot(rentalsWeather, main = "Rentals according to weather", 
        xlab = "Weather situation", ylab = "Total rentals",
        names.arg = c("sunny", "cloudy/rainy", "stormy"))

# Conversion of datatypes
bikes$dteday <- as.Date(as.character(bikes$dteday))
bikesSeason1 <- bikes[bikes$season == 1, ] # needed for anomalies (see below)

bikes$season <- as.factor(bikes$season)
bikes$yr <- as.factor(bikes$yr)
bikes$mnth <- as.factor(bikes$mnth)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$workingday <- as.factor(bikes$workingday)
bikes$weathersit <- as.factor(bikes$weathersit)

# Dummy variables for categorical features
library(fastDummies)
bikes <- dummy_cols(bikes, remove_first_dummy = TRUE)

# Reorder data
bikes <- bikes[c(1, 14:38, 9:13)]

# Look at anomalies
# Left outlier season 4
anomalyDate <- bikes$dteday[bikes$cnt == min(bikes$cnt)] # hurricane in Washington on 2012-10-29
anomalyIndex <- which(bikes$dteday == anomalyDate)
# Replace outlier with mean of the past 30 days
meanMonth <- mean(bikes$cnt[(anomalyIndex-31) : (anomalyIndex-1)])
bikes$cnt[anomalyIndex] <- meanMonth

# Right outlier season 1
anomalyDate <- bikesSeason1$dteday[bikesSeason1$cnt == max(bikesSeason1$cnt)] # St. Patrick's Day
# Don't replace this outlier, since the street parade will take place yearly
# and affect the number of bikes rented
# Same applies for the other very low values in the 2nd year:
# 2012-12-26: Christmas period
# 2012-02-29: last day of February in leap year

# Save number of rows and columns in new data frame
nRows <- nrow(bikes)
nCols <- ncol(bikes)



#------------------------ Splitting the data ------------------------#

library(caTools)
splitBlock <- TRUE 

# Method 1: train on the whole dataset up to the two last 'blocks'
testSize <- 7
cvSize <- 30
if(splitBlock) {
  
  # split data set into consecutive blocks
  trainingSet <- bikes[1 : (nRows-testSize-cvSize), ]
  cvSet <- bikes[(nRows-testSize-cvSize+1) : (nRows-testSize), ]
  trainingCVSet <- bikes[1 : (nRows-testSize), ]
  testSet <- bikes[(nRows-testSize+1) : nRows, ]
  
} else { # Method 2: Choose subsets at random from whole data (60-20-20)
  
  set.seed(123) # fixed randomness
  
  # 60% training - 40% cv+test
  split <- sample.split(bikes$cnt, SplitRatio = 0.6)
  trainingSet <- subset(bikes, split == TRUE)
  cvTestSet <- subset(bikes, split == FALSE)
  
  # 20-20 out of the 40% for cv and test
  split <- sample.split(cvTestSet$cnt, SplitRatio = 0.5)
  cvSet <- subset(cvTestSet, split == TRUE)
  testSet <- subset(cvTestSet, split == FALSE)
}



#------------------------ Applying predictive model ------------------------#
### Time Series ###

library(forecast)
library(TSA)
library(pracma)
library(tseries)

# Number of observations per unit of time (1 = yearly, 12 = monthly)
freq <- 1 

# Convert subsets into time series
tsTrain <- ts(trainingSet$cnt, 
              start = trainingSet$dteday[1], 
              end = trainingSet$dteday[nrow(trainingSet)],
              frequency = freq)
plot(tsTrain, main = "Time Series Train")
abline(reg=lm(tsTrain~time(tsTrain)), col = "red")
tsCV <- ts(cvSet$cnt, 
              start = cvSet$dteday[1], 
              end = cvSet$dteday[nrow(cvSet)],
              frequency = freq)
plot(tsCV, main = "Time Series CV")
tsTrainCV <- ts(trainingCVSet$cnt, 
                start = trainingCVSet$dteday[1], 
                end = trainingCVSet$dteday[nrow(trainingCVSet)],
                frequency = freq)
tsTest <- ts(testSet$cnt, 
              start = testSet$dteday[1], 
              end = testSet$dteday[nrow(testSet)],
              frequency = freq)
plot(tsTest, main = "Time Series Test")


# Make the series stationary
# 1. Deal with the increasing variance -> Log excludes the seasonality
tsTrainLog <- log(tsTrain)
plot(tsTrainLog, main = "Time Series Train (Log)")
abline(reg=lm(tsTrainLog~time(tsTrainLog)), col = "red")


# 2. Mean needs to stay constant over time -> Differentiate to exclude the trend
tsTrainLogDiff <- diff(tsTrainLog)
plot(tsTrainLogDiff, main = "Time Series Train (Log+Differentiated)")
abline(reg=lm(tsTrainLogDiff~time(tsTrainLogDiff)), col = "red")

# Test stationarity
adf.test(tsTrainLogDiff)

# Apply ARIMA Model 
# Auto-Regressive (p) -> predict future values based on past values
# Integrated (d) -> for differentiated non-stationary series 
# Moving-Average (q) -> smoothening process by taking averages of subsets over time

# d = 1, since we only needed to differentiate once to get constant mean
d <- 1

# Auto-correlation function to determine q
acf(tsTrainLogDiff, main = "Auto-correlation")
q <- 2
# Each line is indexed starting from 0
# Take the index of the line before the first inverted line

# Partial auto-correlation function to determine p
pacf(tsTrainLogDiff, main = "Partial auto-correlation")
p <- 16

# Use parameters p, d, q to fit ARIMA-Model
# Note: when specifying d, only the log-Model needs to be passed
# Cross-validation for hyperparameters 'method' and 'transform.pars'
methodCV <- c("CSS-ML", "ML", "CSS")
transPars <- c(TRUE, FALSE)
minError <- 1.e+100
optimalPair <- c("optimalMethod", "optimalTransPars")
for(m in methodCV) {
  for(s in transPars) {
    # this combination yields an error
    if(m == "ML" && s == TRUE) {
      next
    }
    
    # fit ARIMA-Model with current parameter pair
    arimaPredictor <- arima(tsTrainLog, c(p, d, q), 
                            seasonal = list(order = c(p, d, q), period = freq),
                            method = m, transform.pars = s)
    
    # Predict the next perios given by size of cv set
    tsPredLog <- predict(arimaPredictor, n.ahead = cvSize)
    # Predicted values are in log-scale!
    tsPred <- exp(1)^tsPredLog$pred
    
    # Compute error on cv set
    rmseTimeCV <- sqrt(mean((tsPred - tsCV)^2))
    print(paste("Method: ", m, "; Transform_Params: ", s, "; RMSE: ", rmseTimeCV))

    # Update optimal pair of hyperparameters
    if(rmseTimeCV < minError) {
      optimalPair[1] <- m
      optimalPair[2] <- s
      minError <- rmseTimeCV
    }
  }
}

print("Optimal Parameters:")
print(paste("Method: ", optimalPair[1], "; Transform_Params: ", optimalPair[2])) # CSS TRUE
print(paste("Error on CV: ", minError)) # 1468

# Fit model with the optimal hyperparameters on training + cv set
tsTrainCVLog <- log(tsTrainCV)
arimaPredictor <- arima(tsTrainCVLog, c(p, d, q), 
                        seasonal = list(order = c(p, d, q), period = freq),
                        method = optimalPair[1], transform.pars = optimalPair[2])

# Predict last week
tsPredLog <- predict(arimaPredictor, n.ahead = testSize)
# Predicted values are in log-scale!
tsPred <- exp(1)^tsPredLog$pred

# Compute error on test set
rmseTimeTest <- sqrt(mean((tsPred - tsTest)^2)) # 1392

# Plot predicted vs test values 
plot(testSet$dteday, testSet$cnt, type="o", col="red", pch="o", lty=1, ylim=c(0,5000),
     main = "Rentals last week (red = Test, blue = Pred)",
     xlab = "Day", ylab = "Rentals")
points(testSet$dteday, tsPred, col="blue", pch="o")
lines(testSet$dteday, tsPred, col="blue",lty=2)

# Plot prediction attached to original series
ts.plot(tsCV, tsPred, log = "y", lty = c(1, 3), main = "CV set + Prediction last week")


