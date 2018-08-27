dataset = read.csv('/Users/raenug001c/Documents/Machine Learning/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York','California','Florida'),
                         labels = c(1,2,3))

# Splitting the dataset into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)

# Fitting multiple Linear Regression to the Training set
regressor = regressor = lm(formula = Profit ~ .,training_set)
  # or lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
   #  training_set)

# Predicting the Test set results
y_pred = predict(regressor,newdata = test_set)

# Building the optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               dataset)
summary(regressor)

# Removing State variable as it has high P value
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               dataset)
summary(regressor)

# Removing Administration variable as it has high P value
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               dataset)
summary(regressor)

# Removing Marketing.Spend variable as it has high P value
regressor = lm(formula = Profit ~ R.D.Spend,
               dataset)
summary(regressor)

