library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)
library(vroom)

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
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_factor_predictors())

prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazon_train)

#####
## Models
##### 
## logistic regression model
# logRegModel <- logistic_reg() %>% #Type of model
#   set_engine("glm")

## penalized regression model
# penLogModel <- logistic_reg(mixture = tune(), penalty = tune()) %>% 
#   set_engine("glmnet")

## random forest
# forest_mod <- rand_forest(mtry = tune(), 
#                           min_n = tune(), 
#                           trees = 500) %>% 
#   set_engine("ranger") %>% 
#   set_mode("classification")

## knn model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

#####
## Put into a workflow here
#####

## logistic regression
# log_reg_workflow <- workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(logRegModel) %>% 
#   fit(data = amazon_train)

## penalized logistic regression
# amazon_workflow <- workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(penLogModel)

## random forest
# forest_workflow <- workflow() %>% 
#   add_recipe(amazon_recipe) %>% 
#   add_model(forest_mod)

## knn 
knn_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(knn_model)

#####
## CV
#####

## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(), 
#                             mixture(), 
#                             levels = 4)

## Grid for forest
# tuning_grid <- grid_regular(mtry(range = c(1, 9)), 
#                             min_n(), 
#                             levels = 4)

## Grid for knn
tuning_grid <- grid_regular(neighbors(), 
                            levels = 5)

## Split data for CV
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

## Run CV
CV_results <- knn_wf %>% 
  tune_grid(resamples = folds, 
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

## Find best tuning parameters
bestTune <- CV_results %>% 
  select_best(metric="roc_auc")

#####
## Predictions
#####

## Finalize workflow and fit it
final_wf <- knn_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

## Make predictions
amazon_predictions <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob"

## Format predictions
kaggle_predictions <- amazon_predictions %>% 
  bind_cols(., amazon_test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1, 
         Id = id)
  
vroom_write(x = kaggle_predictions, file = "./knnPredictions.csv", delim = ",")
  
