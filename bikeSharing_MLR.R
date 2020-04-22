# Analyzing a bike sharing dataset with multilinear regression
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
### Multilinear Regression ###

# copy of training set for MLR
trainingSetMulti <- trainingSet

# Train multiple linear regressor 
multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
summary(multiRegressor)
# Predictions on CV-set and measure error
predMultiCV <- predict(multiRegressor, newdata = cvSet)
# Compute RMSE between prediction on CV and true outputs from CV
rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) # 1073

# Eliminate insignificant variables 
# Why is workingday NA?
trainingSetMulti <- subset(trainingSetMulti, select = -c(workingday_1)) # drop workingday
multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
summary(multiRegressor)
predMultiCV <- predict(multiRegressor, newdata = cvSet)
rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) # 1073

# Note: dropping further features (even if they are not statistically significant)
#       will increase the error on the cv set

dropInsigFeatures <- FALSE
if(dropInsigFeatures) {
  # Drop atemp (dependent on temp)
  trainingSetMulti <- subset(trainingSetMulti, select = -c(atemp)) 
  multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
  summary(multiRegressor)
  predMultiCV <- predict(multiRegressor, newdata = cvSet)
  rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) # 1074
  
  # Drop dteday
  trainingSetMulti <- subset(trainingSetMulti, select = -c(dteday)) 
  multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
  summary(multiRegressor)
  predMultiCV <- predict(multiRegressor, newdata = cvSet)
  rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) 
  
  # Weekdays are also currently insignificant
  # Keep only the weekend
  trainingSetMulti <- subset(trainingSetMulti, 
                             select = -c(weekday_1, weekday_2, weekday_3, weekday_4)) 
  multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
  summary(multiRegressor)
  predMultiCV <- predict(multiRegressor, newdata = cvSet)
  rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) 
  
  # Weekend still not significant -> drop remaining week days
  trainingSetMulti <- subset(trainingSetMulti, select = -c(weekday_5, weekday_6)) 
  multiRegressor <- lm(formula = cnt ~ ., data = trainingSetMulti)
  summary(multiRegressor)
  predMultiCV <- predict(multiRegressor, newdata = cvSet)
  rmseMultiCV <- sqrt(mean((predMultiCV - cvSet$cnt)^2)) 
}

# Retrain the regressor on the new training + cv set
cvSet <- subset(cvSet, select = -c(workingday_1)) # drop workingday
multiRegressor <- lm(formula = cnt ~ ., data = rbind(trainingSetMulti, cvSet))

# For comparison: error on full training set
rmseMultiTrainCV <- sqrt(mean(multiRegressor$residuals^2)) # 745

# Predict last week
predMultiTest <- predict(multiRegressor, newdata = testSet)

# Compute error on test set
rmseMultiTest <- sqrt(mean((predMultiTest - testSet$cnt)^2)) # 1188

# Plot predicted vs test values 
plot(testSet$dteday, testSet$cnt, type="o", col="red", pch="o", lty=1, ylim=c(0,5000),
     main = "Rentals last week (red = Test, blue = Pred)",
     xlab = "Day", ylab = "Rentals")
points(testSet$dteday, predMultiTest, col="blue", pch="o")
lines(testSet$dteday, predMultiTest, col="blue",lty=2)


