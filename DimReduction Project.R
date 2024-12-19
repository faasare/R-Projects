
stroke.df = healthcare.df
head(stroke.df)
table(stroke.df$stroke)


#1. Data Visualization: Use R with ggplot2 (or other suitable tools) to craft visualizations
# Examining if there is a gender-related trend in stroke occurances
ggplot(stroke.df,
       aes(x = gender, fill = as.factor(stroke))) + 
      geom_bar(position = 'dodge') + 
      labs(title = 'Stroke Counts by Gender', x = 'Gender', fill = 'Stroke')

# Age distribution for people with and without stroke to see if age is a significant factor
ggplot(stroke.df,
       aes(x = age, fill = as.factor(stroke))) + 
       geom_density(alpha = 0.5) + 
       labs(title = 'Age Distribution by Stroke Occurence', x = 'Age', fill = 'Stroke')


# Understanding how glucose level might differ for those who had stroke versus those who did not.
ggplot(stroke.df,
       aes(x = as.factor(stroke), y = avg_glucose_level, fill = as.factor(stroke))) + 
       geom_boxplot() +
       labs(title = "Average Glucose Level by Stroke Status", x = "Stroke", y = "Avg Glucose Level", fill = "Stroke")

# Exploring if certain occupations show higher stroke frequencies
ggplot(stroke.df, 
       aes(x = work_type, fill = as.factor(stroke))) +
       geom_bar(position = "dodge") +
       labs(title = "Work Type by Stroke Occurrence", x = "Work Type", fill = "Stroke")


# Assessing if smoking habits correlate with stroke occurences
ggplot(stroke.df, 
       aes(x = smoking_status, fill = as.factor(stroke))) +
       geom_bar(position = "fill") +
       labs(title = "Smoking Status by Stroke", x = "Smoking Status", y = "Proportion", fill = "Stroke")


# Viewing the overall age distribution and check for skewness
ggplot(stroke.df, 
       aes(x = age)) +
       geom_histogram(bins = 30, fill = "skyblue", color = "black") +
       labs(title = "Age Distribution", x = "Age", y = "Count")



#2. Dimension Reduction Techniques 

# Data Preprocessing
# Converting "NA" string to NA
stroke.df[stroke.df == "N/A"] <- NA
# Checking for missing values
colSums(is.na(stroke.df))


# Using mean of BMI to impute NA values
stroke.df$bmi <- as.numeric(as.character(stroke.df$bmi))
mean_bmi <- mean(stroke.df$bmi, na.rm = TRUE)
stroke.df$bmi[is.na(stroke.df$bmi)] <- mean_bmi
colSums(is.na(stroke.df))


# Convert character columns to factors
stroke.df$gender <- as.factor(stroke.df$gender)
stroke.df$ever_married <- as.factor(stroke.df$ever_married)
stroke.df$work_type <- as.factor(stroke.df$work_type)
stroke.df$Residence_type <- as.factor(stroke.df$Residence_type)
stroke.df$smoking_status <- as.factor(stroke.df$smoking_status)

# Verify changes
str(stroke.df)

# Converting factors to dummy variables
stroke.df.num <- model.matrix(~ . - 1, data = stroke.df)

# Checking the structure of the new data frame
str(data_numeric)

stroke_scaled <- scale(stroke.df.num)


# Perform PCA
pca_result <- prcomp(stroke_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA to see the amount of variance explained by each principal component
summary(pca_result)

# Scree plot to visualize the explained variance
screeplot(pca_result, type = "lines", main = "Scree Plot")




