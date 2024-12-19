# Load the Bike data
Bike <- read.csv("Bike.csv")
head(Bike)

# Load the e1071 library
library(e1071)

# Convert holiday to a factor
Bike$holiday <- as.factor(Bike$holiday)

# Split the data into training and testing sets 2/3 and 1/3
set.seed(123)
train <- sample(1:nrow(Bike), 2/3 * nrow(Bike))

# Train the model
#The response is holiday and the predictors are: season, workingday, casual, and registered. Please use svm() function with radial kernel and gamma=10 and cost = 100.
svm_model <- svm(holiday ~ season + workingday + casual + registered, data = Bike[train,], kernel = "radial", gamma = 10, cost = 100)

# Perform a grid search to find the best model with potential cost: 1, 10, 50, 100 and potential gamma: 1, 3, and 5 and using radial kernel and training dataset.
tune_model <- tune(svm, holiday ~ season + workingday + casual + registered, data = Bike[train,], kernel = "radial", ranges = list(cost = c(1, 10, 50, 100), gamma = c(1, 3, 5)))

# Print the best model
summary(tune_model)

# Forecast holiday using the test dataset and the best model found
pred <- predict(tune_model$best.model, newdata = Bike[-train,])

# Get the true observations of holiday in the test dataset.
trueObservation <- Bike[-train, "holiday"]

# Compute the test error by constructing the confusion matrix. Is it a good model?
table(pred, trueObservation)

# Compute the test error
error <- sum(pred != trueObservation) / length(trueObservation)

# Print the test error
3545/ 3629

