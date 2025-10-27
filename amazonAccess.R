library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)
library(vroom)
# library(reticulate)
# library(keras)
# py_require("tensorflow")
# py_require_legacy_keras()



amazon_train <- vroom("train.csv") %>% 
  mutate(ACTION = factor(ACTION))
amazon_test <- vroom("test.csv") 

#####
### EDA
#####
amazon_train %>% 
  mutate(ROLE_FAMILY = factor(ROLE_FAMILY)) %>% 
  group_by(ROLE_FAMILY) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  head(n = 10) %>% 
ggplot() +
  geom_col(aes(x = ROLE_FAMILY, y = n)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

amazon_train %>% 
  mutate(ROLE_DEPTNAME = factor(ROLE_DEPTNAME)) %>% 
  group_by(ROLE_DEPTNAME) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  head(n = 10) %>% 
ggplot() +
  geom_col(aes(x = ROLE_DEPTNAME, y = n)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#####
### Recipe
##### 
amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_factor_predictors(), threshold = .001) %>%
  #step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>% Target Encoding
  step_dummy(all_factor_predictors())  %>% 
  step_normalize(all_factor_predictors()) %>% 
  step_pca(all_predictors(), threshold = 0.9)

#####
## Neural Network Recipe
#####
# amazon_recipe <- recipe(ACTION ~., data = amazon_train) %>% 
#   update_role(MGR_ID, new_role = "id") %>% 
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_factor_predictors(), threshold = .001) %>%
#   step_dummy(all_factor_predictors()) %>%
#   step_normalize(all_factor_predictors()) %>% 
#   step_range(all_numeric_predictors(), min=0, max=1)
#####

prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazon_train)

#####
## Models
##### 
## logistic regression model
logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

## penalized regression model
# penLogModel <- logistic_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")

## random forest
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

## Naive bayes model
# nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")

## Neural network model
# nn_model <- mlp(hidden_units = tune(), 
#                 epochs = 50) %>% 
#   set_engine("keras") %>% 
#   set_mode("classification")

#####
## Put into a workflow here
#####

## logistic regression
log_reg_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = amazon_train)

## penalized logistic regression
# amazon_workflow <- workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(penLogModel)

## random forest
forest_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(forest_mod)

## knn 
knn_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(knn_model)

## Naive Bayes workflow
# nb_wf <- workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(nb_model)

## Neural Network workflow
# nn_wf <-workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(nn_model)

#####
## CV
#####

## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 4)

## Grid for forest
tuning_grid_forest <- grid_regular(mtry(range = c(1, 9)),
                            min_n(),
                            levels = 4)

## Grid for knn
tuning_grid_knn <- grid_regular(neighbors(),
                            levels = 5)

## Grid for naive bayes
# tuning_grid <- grid_regular(Laplace(), 
#                             smoothness(), 
#                             levels = 4)

## Grid for neural network
# nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)), 
#                             levels = 5)

## Split data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

## Run CV
# CV_results <- amazon_workflow %>% 
#   tune_grid(resamples = folds, 
#             grid = tuning_grid, 
#             metrics = metric_set(roc_auc))

CV_results_forest <- forest_workflow %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid_forest, 
            metrics = metric_set(roc_auc))

CV_results_knn <- knn_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid_knn, 
            metrics = metric_set(roc_auc))

## Find best tuning parameters
# bestTune <- CV_results %>% 
#   select_best(metric="roc_auc")

bestTune_forest <- CV_results_forest %>% 
  select_best(metric="roc_auc")

bestTune_knn <- CV_results_knn %>% 
  select_best(metric="roc_auc")

#####
## Predictions
#####

## Finalize workflow and fit it
# final_wf <- reg_workflow %>% 
#   finalize_workflow(bestTune) %>% 
#   fit(data = amazon_train)

final_wf_forest <- forest_workflow %>% 
  finalize_workflow(bestTune_forest) %>% 
  fit(data = amazon_train)

final_wf_knn <- knn_wf %>% 
  finalize_workflow(bestTune_knn) %>% 
  fit(data = amazon_train)

## Make predictions
amazon_predictions_log <- predict(log_reg_workflow,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob"

amazon_predictions_forest <- predict(final_wf_forest,
                                  new_data=amazon_test,
                                  type="prob") # "class" or "prob"

amazon_predictions_knn <- predict(final_wf_knn,
                                  new_data=amazon_test,
                                  type="prob") # "class" or "prob"

## Format predictions
kaggle_predictions_log <- amazon_predictions_log %>% 
  bind_cols(., amazon_test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1, 
         Id = id)

kaggle_predictions_forest <- amazon_predictions_forest %>% 
  bind_cols(., amazon_test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1, 
         Id = id)

kaggle_predictions_knn <- amazon_predictions_knn %>% 
  bind_cols(., amazon_test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1, 
         Id = id)
  
vroom_write(x = kaggle_predictions_log, file = "./logPredictionsPCA.csv", delim = ",")
vroom_write(x = kaggle_predictions_forest, file = "./forestPredictionsPCA.csv", delim = ",")
vroom_write(x = kaggle_predictions_knn, file = "./knnPredictionsPCA.csv", delim = ",")
  
## Create tuning graphic
# CV_results %>% collect_metrics() %>% 
#   filter(.metric=="roc_auc") %>% 
#   ggplot(aes(x = hidden_units, y = mean)) +
#   geom_line()
