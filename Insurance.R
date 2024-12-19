#############################################
#                                           #
# Author:     Felix Asare                 #
# Date:       Date                          #
# Subject:    Final Project                 #
# Class:      BDAT 640                      #
# Section:    01W                           #         
# Instructor: Chris Shannon                 #
# File Name:  FinalProject_Asare_Felix.R    #
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################

#     a.  Load the dataset insurance.csv into memory.
head(insurance)
#     b.  In the data frame, transform the variable charges by seting
#         insurance$charges = log(insurance$charges). Do not transform
#         it outside of the data frame.
insurance$charges <- log(insurance$charges)

#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.
insurance_model <- model.matrix( ~ ., data = insurance)
insurance_model <- insurance_model[, -1]
head(insurance_model)

#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.
index <- sample(1:nrow(insurance_model), floor(2/3 * nrow(insurance_model)))

#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.
train_e <- insurance[index, ]
test_e <- insurance[-index, ]

#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d
train_f <- insurance_model[index,]
test_f <- insurance_model[-index,]

#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the training data set created in
#         step 1.e to train your model.
lm.model <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train_e)
summary(lm.model)


#     b.  Is there a relationship between the predictors and the
#         response?
#     c.  Does sex have a statistically significant relationship to the
#         response?
#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direciton="backward"
library(MASS)
step.model <- stepAIC(lm.model, direction = "backward")

#     e.  Compute the test error of the best model in #3d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.
library(caret)
ctrl <- trainControl(method = "LOOCV")
lm.model <- train(charges ~ age + bmi + smoker, data = train_e, method = "lm", trControl = ctrl)
print(lm.model)
0.4580681 ^ 2

#     f.  Calculate the test error of the best model in #3d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #3d based
#         on AIC. Report the MSE.
ctrl <- trainControl(method = "cv", number = 10)
cv.model <- train(charges ~ age + bmi + smoker, data = train_e, method = "lm", trControl = ctrl)
print(cv.model)
aic_mse = (cv.model$results$RMSE)^2
aic_mse

#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.
predictions <- predict(step.model, newdata = test_e)
actual_values <- test_e$charges
best_lm_mse <- mean((predictions - actual_values)^2)
cat("Test MSE:", best_lm_mse)


#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?
print(aic_mse - best_lm_mse)

######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.
library(tree)
tree_model <- tree(charges ~ age + sex + bmi + children + smoker + region, 
                   data = train_e)

#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.
cv_tree <- cv.tree(tree_model)
plot(cv_tree$size, cv_tree$dev, type = "b",
     xlab = "Tree Size (Number of Terminal Nodes)",
     ylab = "Deviance",
     main = "Cross-Validation for Optimal Tree Size")
best_size <- cv_tree$size[which.min(cv_tree$dev)]
cat("Best tree size is ", best_size)

#     c.  Justify the number you picked for the optimal tree with
#         regard to the principle of variance-bias trade-off.
print("Explanation in word doc")
#     d.  Prune the tree using the optinal size found in 3.b
prune_tree <- prune.tree(tree_model, best = best_size)


#     e.  Plot the best tree model and give labels.
plot(prune_tree)
text(prune_tree, pretty = 0)
#     f.  Calculate the test MSE for the best model.
predictions <- predict(prune_tree, newdata = test_e)
actual_values <- test_e$charges
tree_mse <- mean((predictions - actual_values)^2)
cat("Test MSE:", tree_mse)


####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.
library(randomForest)
rf_model <- randomForest(charges ~ age + sex + bmi + children + smoker + region, 
                         data = train_e, 
                         importance = TRUE)

#     b.  Compute the test error using the test data set.
yhat.rf <- predict(rf_model, newdata = test_e)
actual_values <- test_e$charges
rf_te = mean((yhat.rf - actual_values)^2)
rf_te

#     c.  Extract variable importance measure using the importance()
#         function.
importance(rf_model)


#     d.  Plot the variable importance using the function, varImpPlot().
varImpPlot(rf_model)
#         Which are the top 3 important predictors in this model?
print("Top 3 important predictors are smoker, age, and children")

############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.
library(e1071)
svm_model <- svm(charges ~ age + sex + bmi + children + smoker + region, 
                 data = train_e, 
                 kernel = "radial", 
                 gamma = 5, 
                 cost = 50)
#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","polynomial","radial" and
#         "sigmoid". And use the training set created in step 1.e.
tune.out <- tune(svm, 
                 charges ~ age + sex + bmi + children + smoker + region,
                 data = train_e,
                 ranges = list(cost = c(1, 10, 50, 100),
                               gamma = c(1, 3, 5)),
                 kernel = c("radial"))


#     c.  Print out the model results. What are the best model
#         parameters?
summary(tune.out)

#     d.  Forecast charges using the test dataset and the best model
#         found in c).
pred = predict(tune.out$best.model, newdata = test_e)

#     e.  Compute the MSE (Mean Squared Error) on the test data.
actual_values = test_e$charges
svm_mse <- mean((pred - actual_values)^2)
cat("Mean Squared Error", svm_mse)

#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#     a.  Use the training data set created in step 1.f and standardize
#         the inputs using the scale() function.
scaled_df <- scale(train_f)
head(scaled_df)

#     b.  Convert the standardized inputs to a data frame using the
#         as.data.frame() function.
scaled_df <- as.data.frame(scared_df)


#     c.  Determine the optimal number of clusters, and use the
#         gap_stat method and set iter.max=20. Justify your answer.
#         It may take longer running time since it uses a large dataset.
library(cluster)
library(factoextra)
fviz_nbclust(scaled_df, kmeans, method = "gap_stat", iter.max = 20)


#     d.  Perform k-means clustering using the optimal number of
#         clusters found in step 6.c. Set parameter nstart = 25
km.res <- kmeans(scaled_df, 7, nstart = 25)


#     e.  Visualize the clusters in different colors, setting parameter
#         geom="point"
fviz_cluster(km.res, 
             data = scaled_df,
             geom = "point", 
             stand = FALSE, 
             ellipse = TRUE, 
             show.clust.cent = TRUE,
             palette = "jco") 
######################################
# 7.  Build a neural networks model. #
######################################

#     a.  Using the training data set created in step 1.f, create a 
#         neural network model where the response is charges and the
#         predictors are age, sexmale, bmi, children, smokeryes, 
#         regionnorthwest, regionsoutheast, and regionsouthwest.
#         Please use 1 hidden layer with 1 neuron. Do not scale
#         the data.
library(neuralnet)
# head(train_f)
train_f_scaled <- train_f
train_f_scaled <- as.data.frame(train_f_scaled)
nn_model <- neuralnet(charges ~ age + sexmale + bmi + children + smokeryes + 
                        regionnorthwest + regionsoutheast + regionsouthwest, 
                      data = train_f_scaled, 
                      hidden = 1)

#     b.  Plot the neural network.
plot(nn_model)
#     c.  Forecast the charges in the test dataset.
scaled_test_f <- scale(test_f)
scaled_test_f <- as.data.frame(scaled_test_f)
predictions <- compute(nn_model, scaled_test_f[, c("age", "sexmale", "bmi", "children", "smokeryes", 
                                                   "regionnorthwest", "regionsoutheast", "regionsouthwest")
])
#     d.  Compute test error (MSE).
observ_test <- scaled_test_f$charges
nn_mse <- mean((observ_test - predictions$net.result)^2)
nn_mse

################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.d. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.

mse_multiple_regression <- 0.2079   
mse_regression_tree <- 0.5973       
mse_random_forest <- 0.1546          
mse_svm <- 0.2096                   
mse_neural_network <- 52.2545    
results_df <- data.frame(
  Model.Type = c("Multiple Linear Regression", "Regression Tree", 
                 "Random Forest", "Support Vector Machine", "Neural Network"),
  Test.MSE = c(mse_multiple_regression, mse_regression_tree, 
               mse_random_forest, mse_svm, mse_neural_network)
)

results_df$Test.MSE <- round(results_df$Test.MSE, 4)
print(results_df)

best_model <- results_df[which.min(results_df$Test.MSE), ]
print(paste("The best model is:", best_model$Model.Type, "with a Test MSE of", best_model$Test.MSE))


#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?
# View feature importance
library(randomForest)
importance(rf_model)

# Plot feature importance
varImpPlot(rf_model, 
           main = "Variable Importance in Random Forest Model",
           n.var = min(10, nrow(importance(rf_model)))) 
importance_values <- randomForest::importance(rf_model)
importance_values




#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result.
#         Follow these steps:
#
#         i.   Copy your pruned tree model to a new variable.
copy_of_my_pruned_tree <- prune_tree


#         ii.  In your new variable, find the data.frame named
#              "frame" and reverse the log transformation on the
#              data.frame column yval using the exp() function.
#              (If the copy of your pruned tree model is named 
#              copy_of_my_pruned_tree, then the data frame is
#              accessed as copy_of_my_pruned_tree$frame, and it
#              works just like a normal data frame.).
copy_of_my_pruned_tree$frame$yval <- exp(copy_of_my_pruned_tree$frame$yval)


#         iii. After you reverse the log transform on the yval
#              column, then replot the tree with labels.
plot(copy_of_my_pruned_tree)
text(copy_of_my_pruned_tree, pretty = 0, cex = 0.8)


