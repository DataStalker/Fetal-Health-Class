# Load necessary packages
library(tidyverse)   # For data manipulation and visualization
library(tidymodels)  # For modeling
library(vip)         # For variable importance
library(corrplot)    # For correlation matrix visualization
library(yardstick)   # For confusion matrix visualization
library(xgboost)     # For XGBoost model


### 2. Load and Prepare the Data


# Load the data
df <- read.csv('fetal_health.csv')

# Convert the target variable to a factor for classification
df$fetal_health <- as.factor(df$fetal_health)


### 3. Exploratory Data Analysis (EDA)

#### Target Distribution


# Plot the distribution of the target variable
df %>%
  ggplot(aes(x = fetal_health)) +
  geom_bar(fill = 'steelblue') +
  labs(title = 'Target Distribution', x = 'Fetal Health', y = 'Count') +
  theme_minimal()


#### Correlation Matrix


# Calculate the correlation matrix (excluding the target variable)
correlation_matrix <- df %>%
  select(-fetal_health) %>%
  cor(method = "spearman")

# Adjust margins to fit the plot in the window
par(mar = c(1, 1, 1, 1))

# Plot the correlation matrix
corrplot(correlation_matrix, method = "color", tl.cex = 0.6)


### 4. Feature Distribution Against the Outcome

#### Density Plots for Continuous Features


# Reshape data for easy plotting
df_long <- df %>%
  pivot_longer(-fetal_health, names_to = "features", values_to = "value")

# Plot density plots for each feature by fetal health
df_long %>%
  ggplot(aes(x = value, fill = fetal_health)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ features, scales = "free") +
  labs(title = "Feature Distributions by Fetal Health Outcome", x = "Value", y = "Density") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")


#### Box Plots for Continuous Features


# Plot boxplots for each feature by fetal health
df_long %>%
  ggplot(aes(x = fetal_health, y = value, fill = fetal_health)) +
  geom_boxplot() +
  facet_wrap(~ features, scales = "free") +
  labs(title = "Feature Distributions by Fetal Health Outcome", x = "Fetal Health", y = "Value") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set2")


### 5. Data Splitting


# Set seed for reproducibility
set.seed(142)

# Split data into training (70%) and testing (30%) sets
df_split <- initial_split(df, prop = 0.7, strata = fetal_health)
df_train <- training(df_split)
df_test <- testing(df_split)


### 6. Preprocessing with `recipes`

# Create a recipe for data preprocessing
fetal_health_recipe <- recipe(fetal_health ~ ., data = df_train) %>%
  step_normalize(all_numeric_predictors())        

# Prepare the recipe
fetal_health_recipe <- fetal_health_recipe %>% prep()


### 7. Model Specification

#### XGBoost


# Define the XGBoost model with tuning
xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), 
  learn_rate = tune(), 
  loss_reduction = tune(),
  sample_size = tune(), 
  mtry = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Create a workflow for XGBoost
xgb_wf <- workflow() %>%
  add_recipe(fetal_health_recipe) %>%
  add_model(xgb_spec)


### 8. Model Tuning and Fitting

#### Tuning XGBoost


# Set up cross-validation and tune the XGBoost model
set.seed(142)
xgb_res <- xgb_wf %>%
  tune_grid(
    resamples = vfold_cv(df_train, v = 5),
    grid = 20,
    metrics = metric_set(roc_auc)
  )

# Select the best model based on ROC AUC
best_xgb <- xgb_res %>%
  select_best("roc_auc")

# Finalize the workflow with the best hyperparameters
final_xgb <- finalize_workflow(xgb_wf, best_xgb)

# Fit the final XGBoost model on the training data
final_xgb_fit <- final_xgb %>%
  fit(data = df_train)

### 9. Model Evaluation

#### XGBoost Evaluation


# Make predictions on the test set using XGBoost
xgb_predictions <- final_xgb_fit %>%
  predict(df_test) %>%
  bind_cols(df_test)

# Calculate evaluation metrics for XGBoost
xgb_metrics <- xgb_predictions %>%
  metrics(truth = fetal_health, estimate = .pred_class)

# Print the evaluation metrics for XGBoost
xgb_metrics


### 10. Confusion Matrix with Plot

#### GLM Confusion Matrix

#### XGBoost Confusion Matrix


# Create and plot the confusion matrix for XGBoost
xgb_conf_mat <- xgb_predictions %>%
  conf_mat(truth = fetal_health, estimate = .pred_class)

# Visualize the confusion matrix for XGBoost
xgb_conf_mat %>%
  autoplot(type = "heatmap") +
  labs(title = "Confusion Matrix: XGBoost Model")


### 11. Variable Importance


# Plot variable importance for the XGBoost model
final_xgb_fit %>%
  extract_fit_parsnip() %>%
  vip(geom = "point") +
  labs(title = "Variable Importance: XGBoost Model")


### 12. ROC Curve for Model Performance

#### XGBoost ROC Curve


# Get predicted probabilities for XGBoost
xgb_prob_predictions <- final_xgb_fit %>%
  predict(df_test, type = "prob") %>%
  bind_cols(df_test)

# Calculate ROC curve for XGBoost
xgb_prob_predictions %>%
  roc_curve(truth = fetal_health, .pred_1:.pred_3) %>%  # Adjust according to your factor levels
  autoplot() +
  labs(title = "ROC Curve: XGBoost Model")
