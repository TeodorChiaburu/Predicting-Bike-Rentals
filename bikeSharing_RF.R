# Analyzing a bike sharing dataset with random forests
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
### Random Forests ###


# Define function for optimizing #trees, #variables to consider at each split
# and maximal #observations in terminal nodes
rfGridSearch <- function(formula, data, numTrees, splitVars, nodeSizes) {
  
  library(randomForest)
  optimalParams <- c(500, 10, 5) # default values (10 = nCols/3)
  minErrorCV <- 1.e+100
  set.seed(123)
  
  for(t in numTrees) {
    for(s in splitVars) {
      for(n in nodeSizes) {
        
        # fit model with current parameters
        forestRegressor <- randomForest(formula = formula, data = data, 
                                        ntree = n, mtry = s, nodesize = n)
        
        # Predictions on CV-set and measure error
        predForestCV <- predict(forestRegressor, newdata = cvSet)
        
        # Compute RMSE between prediction on CV and true outputs from CV
        rmseForestCV <- sqrt(mean((predForestCV - cvSet$cnt)^2))
        print(paste("Trees: ", t, "; mtry: ", s, " nodesize: ", n,"; RMSE CV: ", rmseForestCV))
        
        # Update optimal hyperparameters
        if(rmseForestCV < minErrorCV) {
          optimalParams[1] <- t
          optimalParams[2] <- s
          optimalParams[3] <- n
          minErrorCV <- rmseForestCV
        }
      }
    }
  }
  
  print("Optimal Parameters:")
  print(paste("Trees: ", optimalParams[1], "; mtry: ", optimalParams[2], " nodesize: ", optimalParams[3]))
  print(paste("Error on CV: ", minErrorCV))
  
  return(optimalParams)
}

# Grid search
numTrees <- c(500, 1000, 1500) 
splitVars <- c(10, 15, 20) 
nodeSizes <- c(5, 10, 20) 
optimalParams <- rfGridSearch(formula = trainingSet$cnt ~ ., data = trainingSet, 
                              numTrees, splitVars, nodeSizes)
# Optimal Parameters: 1500 trees, 10 split variables, node size = 5
# Minimal CV Error: 1036

# Try more trees, mtry between 5 and 15, node size between 1 and 10
numTrees <- c(1500, 2000, 2500, 3000)
splitVars <- seq(5, 15, by = 1)
nodeSizes <- seq(1, 10, by = 1)
optimalParams <- rfGridSearch(formula = trainingSet$cnt ~ ., data = trainingSet, 
                              numTrees, splitVars, nodeSizes)
# Optimal Parameters: 1500 trees, 6 split variables, node size = 8
# Minimal CV Error: 869


# Fit model with the optimal hyperparameters on training + cv set
set.seed(123)
forestRegressor <- randomForest(x = trainingCVSet[-nCols], y = trainingCVSet$cnt, 
                                ntree = optimalParams[1], 
                                mtry = optimalParams[2], 
                                nodesize = optimalParams[3])

# For comparison: error on full training set
rmseForestTrainCV <- sqrt(mean(forestRegressor$predicted - trainingCVSet$cnt)^2) # 6

# Predict last week
predForestTest <- predict(forestRegressor, newdata = testSet)

# Compute error on test set
rmseForestTest <- sqrt(mean((predForestTest - testSet$cnt)^2)) # 1368

# Plot predicted vs test values 
plot(testSet$dteday, testSet$cnt, type="o", col="red", pch="o", lty=1, ylim=c(0,5000),
     main = "Rentals last week (red = Test, blue = Pred)",
     xlab = "Day", ylab = "Rentals")
points(testSet$dteday, predForestTest, col="blue", pch="o")
lines(testSet$dteday, predForestTest, col="blue", lty=2)


