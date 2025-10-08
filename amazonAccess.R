library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)

amazon_train <- read_csv("train.csv")
amazon_test <- read_csv("test.csv")

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
  step_dummy(all_factor_predictors()) %>% 
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION))

prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazon_train)
