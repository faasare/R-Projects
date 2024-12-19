# Data Understanding
# Data Dimension
dim(stroke.df)

# Head of data
head(stroke.df)
stroke.df

#Data types
str(stroke.df)

library(dplyr)

# Convert bmi from character to numeric, treating "N/A" as NA
stroke.df$bmi <- as.numeric(gsub("N/A", NA, stroke.df$bmi))

# Check for any other columns that need type correction
stroke.df$age <- as.numeric(stroke.df$age)
stroke.df$avg_glucose_level <- as.numeric(stroke.df$avg_glucose_level)

# Summary to check for missing values
summary(stroke.df)

# Impute missing values for bmi with the median (excluding NA values)
median_bmi <- median(stroke.df$bmi, na.rm = TRUE)
stroke.df$bmi[is.na(stroke.df$bmi)] <- median_bmi

#Convert Categorical Variables to Factors
stroke.df$gender <- as.factor(stroke.df$gender)
stroke.df$ever_married <- as.factor(stroke.df$ever_married)
stroke.df$work_type <- as.factor(stroke.df$work_type)
stroke.df$Residence_type <- as.factor(stroke.df$Residence_type)
stroke.df$smoking_status <- as.factor(stroke.df$smoking_status)
# Impute missing values for bmi using the median
median_bmi <- median(stroke.df$bmi, na.rm = TRUE)
stroke.df$bmi[is.na(stroke.df$bmi)] <- median_bmi

# Verify changes
summary(stroke.df$bmi)

# Plotting histograms for age, avg_glucose_level, and bmi
library(ggplot2)

# Age Distribution
ggplot(stroke.df, aes(x=age)) + 
  geom_histogram(bins=30, fill="skyblue", color="black") +
  ggtitle("Distribution of Age") +
  xlab("Age") +
  ylab("Frequency")

# Average Glucose Level Distribution
ggplot(stroke.df, aes(x=avg_glucose_level)) +
  geom_histogram(bins=30, fill="lightgreen", color="black") +
  ggtitle("Distribution of Average Glucose Level") +
  xlab("Average Glucose Level") +
  ylab("Frequency")

# BMI Distribution
ggplot(stroke.df, aes(x=bmi)) +
  geom_histogram(bins=30, fill="salmon", color="black") +
  ggtitle("Distribution of BMI") +
  xlab("BMI") +
  ylab("Frequency")


# Gender Distribution
ggplot(stroke.df, aes(x=gender)) +
  geom_bar(fill="lightblue") +
  ggtitle("Gender Distribution") +
  xlab("Gender") +
  ylab("Count")

# Smoking Status Distribution
ggplot(stroke.df, aes(x=smoking_status)) +
  geom_bar(fill="lightgreen") +
  ggtitle("Smoking Status Distribution") +
  xlab("Smoking Status") +
  ylab("Count")

# Work Type Distribution
ggplot(stroke.df, aes(x=work_type)) +
  geom_bar(fill="pink") +
  ggtitle("Work Type Distribution") +
  xlab("Work Type") +
  ylab("Count")

# Boxplot for BMI
ggplot(stroke.df, aes(x="", y=bmi)) +
  geom_boxplot(fill="tan") +
  ggtitle("Boxplot of BMI")

# Boxplot for Average Glucose Level
ggplot(stroke.df, aes(x="", y=avg_glucose_level)) +
  geom_boxplot(fill="orange") +
  ggtitle("Boxplot of Average Glucose Level")


# Scatter Plot for Age vs. Avg Glucose Level colored by Stroke Outcome
ggplot(stroke.df, aes(x=age, y=avg_glucose_level, color=factor(stroke))) +
  geom_point(alpha=0.6) +
  ggtitle("Age vs. Average Glucose Level by Stroke Outcome") +
  xlab("Age") +
  ylab("Average Glucose Level") +
  scale_color_manual(values=c("red", "green"), labels=c("No Stroke", "Stroke"))


# Box plot of Age vs Stroke
ggplot(stroke.df, aes(x=factor(stroke), y=age, fill=factor(stroke))) +
  geom_boxplot() +
  labs(x="Stroke", y="Age", title="Distribution of Age by Stroke Incidence") +
  scale_fill_manual(values=c("red", "green"), labels=c("Stroke", "No Stroke")) +
  theme_minimal()

stroke.df$stroke <- factor(stroke.df$stroke, levels = c(0, 1), labels = c("No Stroke", "Stroke"))

# Creating a grouped bar chart for Work Type vs. Stroke Incidence
ggplot(stroke.df, aes(x = work_type, fill = stroke)) +
  geom_bar(position = "dodge") +
  labs(x = "Work Type", y = "Count", title = "Distribution of Stroke Incidence by Work Type") +
  scale_fill_brewer(palette = "Set1", name = "Stroke Incidence") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

table(stroke.df$stroke)  # Check balance


# Load necessary libraries
library(rpart)
library(caret)  
install.packages("ROSE")
library(ROSE)

# Applying the ROSE method to generate synthetic data
set.seed(123)
# Calculate total needed samples based on the desired balance
total_strokes <- sum(stroke.df$stroke == "Stroke")
total_no_strokes <- nrow(stroke.df) - total_strokes
desired_total <- 2 * max(total_strokes, total_no_strokes)

# Applying ROSE with adjusted N
rose_data <- ovun.sample(stroke ~ ., data = stroke.df, method = "over", N = desired_total)$data

# Check the new balance of classes
table(rose_data$stroke)

index <- createDataPartition(rose_data$stroke, p = 0.80, list = FALSE)
train_data_balanced <- rose_data[index, ]
test_data_balanced <- rose_data[-index, ]

# Build the decision tree model
tree.model <- rpart(stroke ~ . - id, data = train_data_balanced, method = "class",
                    control = rpart.control(minsplit = 20, minbucket = 7, 
                                            maxdepth = 10, usesurrogate = 2, xval =10 ))

# Plot the decision tree
plot(tree.model, main = "Decision Tree for Stroke Prediction on Balanced Data")
text(tree.model, use.n = TRUE)

# Evaluation
library(pROC)
predictions <- predict(tree.model, newdata = test_data_balanced, type = "class")
prob_predictions <- predict(tree.model, newdata = test_data_balanced, type = "prob")[, "Stroke"]


conf_mat <- confusionMatrix(predictions, test_data_balanced$stroke)

# Extracting metrics from confusion matrix
accuracy <- conf_mat$overall['Accuracy']
precision <- conf_mat$byClass['Precision']
recall <- conf_mat$byClass['Sensitivity']
f1_score <- 2 * (precision * recall) / (precision + recall) 
specificity <- conf_mat$byClass['Specificity']

# Print metrics
cat("Accuracy:", accuracy, "\n",
    "Precision:", precision, "\n",
    "Recall (Sensitivity):", recall, "\n",
    "F1 Score:", f1_score, "\n",
    "Specificity:", specificity, "\n")

roc_result <- roc(response = test_data_balanced$stroke, predictor = as.numeric(prob_predictions))
plot(roc_result, main = "ROC Curve")
auc_value <- auc(roc_result)
cat("AUC:", auc_value, "\n")

# Building Logistic Regression model.
# Convert categorical variables into dummy variables using model.matrix
dummy_data <- model.matrix(~ gender + work_type + smoking_status - 1, data = rose_data)

# Combine dummy variables with the rest of the dataset
rose_data <- cbind(rose_data[, !(names(rose_data) %in% c("gender", "work_type", "smoking_status"))], dummy_data)

#Removing id
rose_data <- rose_data[, !names(rose_data) %in% c("id")]

# Scaling Data
preProcValues <- preProcess(rose_data[, c("age", "avg_glucose_level", "bmi")], method = c("center", "scale"))
rose_data_scaled <- predict(preProcValues, rose_data)

# Verify the structure of the prepared dataset
str(rose_data_scaled)

# Split into training and testing datasets
set.seed(123)
index <- createDataPartition(rose_data_scaled$stroke, p = 0.80, list = FALSE)
train_data <- rose_data_scaled[index, ]
test_data <- rose_data_scaled[-index, ]

# Fit the logistic regression model
logistic_model <- glm(stroke ~ ., data = train_data, family = binomial())

# View the model summary
summary(logistic_model)

# Predict probabilities on the test data
predicted_probs <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions
predicted_class <- ifelse(predicted_probs > 0.5, "Stroke", "No Stroke")
predicted_class <- factor(predicted_class, levels = c("No Stroke", "Stroke"))

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_class, test_data$stroke)

# Print evaluation metrics
accuracy <- conf_matrix$overall['Accuracy']
precision <- conf_matrix$byClass['Precision']
recall <- conf_matrix$byClass['Sensitivity']
f1_score <- 2 * (precision * recall) / (precision + recall)  # F1 Score
specificity <- conf_matrix$byClass['Specificity']

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Specificity:", specificity, "\n")

# Generate ROC curve and calculate AUC
roc_curve <- roc(test_data$stroke, predicted_probs)
plot(roc_curve, main = "ROC Curve for Logistic Regression")
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

# Building Neural Network
# Install neuralnet package if not already installed
if (!require("neuralnet")) install.packages("neuralnet")
library(neuralnet)

# Convert stroke variable to numeric binary
rose_data_scaled$stroke <- ifelse(rose_data_scaled$stroke == "Stroke", 1, 0)

rose_data_scaled$ever_married <- ifelse(rose_data_scaled$ever_married == "Yes", 1, 0)
rose_data_scaled$Residence_type <- ifelse(rose_data_scaled$Residence_type == "Urban", 1, 0)

# Normalize all numeric columns except the target variable
numeric_columns <- names(rose_data_scaled)[!names(rose_data_scaled) %in% c("stroke")]
rose_data_scaled[numeric_columns] <- scale(rose_data_scaled[numeric_columns])

set.seed(123)
index <- createDataPartition(rose_data_scaled$stroke, p = 0.80, list = FALSE)
train_data <- rose_data_scaled[index, ]
test_data <- rose_data_scaled[-index, ]

# Clean column names
names(train_data) <- make.names(names(train_data))
names(test_data) <- make.names(names(test_data))

# Create the formula dynamically
predictors <- names(train_data)[!names(train_data) %in% c("stroke")]
formula <- as.formula(paste("stroke ~", paste(predictors, collapse = " + ")))
print(formula)

# Train the neural network
library(neuralnet)
nn_model <- neuralnet(formula, 
                      data = train_data, 
                      hidden = c(5, 3),    
                      linear.output = FALSE, 
                      lifesign = "minimal",
                      stepmax = 1e+06)

# Plot the neural network
plot(nn_model)

# Make predictions on the test data
nn_predictions <- compute(nn_model, test_data[, predictors])

# Extract predicted probabilities
predicted_probs <- nn_predictions$net.result

# Convert probabilities to binary predictions (threshold = 0.5)
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)
predicted_class <- as.factor(predicted_class)

# Confusion matrix
conf_matrix <- confusionMatrix(predicted_class, as.factor(test_data$stroke))

# Print confusion matrix and evaluation metrics
print(conf_matrix)

# Extract key metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]  
f1_score <- 2 * (precision * recall) / (precision + recall) 
specificity <- conf_matrix$byClass["Specificity"]

cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Specificity:", specificity, "\n")

# Generate ROC curve
roc_curve <- roc(as.numeric(test_data$stroke), as.numeric(predicted_probs))

# Plot the ROC curve
plot(roc_curve, main = "ROC Curve for Neural Network", col = "blue", lwd = 2)
abline(a = 0, b = 1, col = "gray", lty = 2)  

# Calculate AUC
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

