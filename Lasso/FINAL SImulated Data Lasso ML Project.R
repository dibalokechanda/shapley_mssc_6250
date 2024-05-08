# Clear workspace
rm(list=ls())

# Set random seed
set.seed(6520)

# Load Libraries
library(glmnet)

# Specify the mean vector 
mu <- numeric(10)

# Specify the value for the correlation in the covariance matrix
rho<-0.05

# Generate the covariance  matrix
Sigma <- matrix(rho, nrow = 10, ncol = 10)
diag(Sigma) <- 1

# Generate input values for training and test from multivariate Gaussian distribution
train_x_matrix <- MASS::mvrnorm(n = 1000, mu, Sigma)
test_x_matrix <- MASS::mvrnorm(n = 500, mu, Sigma)

# Generate noise samples 
train_sample_noises <- rnorm(1000, mean = 0, sd = 1)
test_sample_noises <- rnorm(500, mean = 0, sd = 1)

# Generate beta vector
beta_2 <- numeric(5) 
beta_1 <- c(0.6, 0.6, -0.2, -0.05, 1)
beta_vec <- c(beta_1, beta_2)

# Generate training output values
train_y<- train_x_matrix %*% beta_vec+train_sample_noises

# Generate test output values
test_y<- test_x_matrix %*% beta_vec+test_sample_noises

# Fitting Lasso with 10-fold cross-validation
lasso_fit <- cv.glmnet(x = train_x_matrix, y =train_y, nfolds = 10, 
                       alpha = 1,intercept=FALSE)

# Visualize
plot(lasso_fit$glmnet.fit, "lambda")
plot(lasso_fit)

# Get the optimal value for lambda 
opt_lambda_min <- lasso_fit$lambda.min

# See the coefficients
coef(lasso_fit, s = "lambda.1se")

# Perform prediction on the test data
lasso_pred <- predict(lasso_fit,newx=test_x_matrix)

# Compute the test MSE
mean((lasso_pred-test_y)^2)