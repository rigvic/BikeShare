library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)



# LINEAR REGRESSION
bike_train <- vroom("./train.csv")
bike_test <- vroom("./test.csv")

# cleaning
bike_train <- bike_train %>%
  select(-casual, -registered)

# feature engineering
my_recipe <- recipe(count~., data = bike_train) %>%
  step_num2factor(season, levels = c("Spring", "Summer", "Fall", "Winter")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_num2factor(weather, levels = c("Clear", "Mist", "Rain")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime)


#my_recipe <- prep(my_recipe)
prepped_recipe <- prep(my_recipe)
bake(my_recipe, new_data = bike_train)
baked_recipe <- bake(my_recipe, new_data = bike_test)

extract_fit_engine(bike_workflow) %>%
  tidy()
extract_fit_engine(bike_workflow) %>%
  summary()

my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_train) # Fit the workflow
bike_predictions <- predict(bike_workflow,
                            new_data=bike_test) # Use fit to predict

bike_predictions$count <- ifelse(bike_predictions$.pred <= 0, 0, bike_predictions$.pred)
bike_predictions$datetime <- as.character(format(bike_test$datetime))
bike_predictions <- bike_predictions %>% select(datetime, count)

vroom_write(x=bike_predictions, file="bike_predictions.csv", delim=",")

# POISSON REGRESSION
pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_train) # Fit the workflow

bike_predictions_pois <- predict(bike_pois_workflow,
                                 new_data= bike_test) # Use fit to predict

bike_predictions_pois$count <- bike_predictions_pois$.pred

bike_predictions_pois$datetime <- as.character(format(bike_test$datetime))
bike_predictions_pois <- bike_predictions_pois %>% select(datetime, count)

vroom_write(x=bike_predictions_pois, file="bike_predictions_pois.csv", delim=",")

# LOG RECIPE
my_recipe2 <- recipe(count~., data = bike_train) %>%
  step_num2factor(season, levels = c("Spring", "Summer", "Fall", "Winter")) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>%
  step_num2factor(weather, levels = c("Clear", "Mist", "Rain")) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = "year") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

log_train_set <- bike_train %>%
  mutate(count=log(count))

lin_model <- linear_reg() %>%
  set_engine("lm")


preg_model <- poisson_reg(penalty = .25, mixture = .115) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model) %>%
  fit(data=log_train_set)

bike_predictions_preg <- predict(preg_wf, new_data = bike_test) %>%
  mutate(.pred = exp(.pred))

bike_predictions_preg$count <- bike_predictions_preg$.pred
bike_predictions_preg$datetime <- as.character(format(bike_test$datetime))
bike_predictions_preg <- bike_predictions_preg %>% select(datetime, count)

vroom_write(x=bike_predictions_preg, file="bike_predictions_preg.csv", delim=",")

preg_model_2 <- linear_reg(penalty = tune(),
                           mixture = tune()) %>%
  set_engine("glmnet")

preg_wf_2 <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model_2)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(log_train_set, v = 5, repeats = 1)

CV_results <- preg_wf_2 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data =., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- preg_wf_2 %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_train_set)

# bike_predictions_preg_2 <- predict(preg_wf_2, new_data = bike_test) %>%
# mutate(.pred = exp(.pred))

bike_predictions_preg_2 <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))


# bike_predictions_preg_2$count <- bike_predictions_preg_2$.pred
# bike_predictions_preg_2$datetime <- as.character(format(bike_test$datetime))
# bike_predictions_preg_2 <- bike_predictions_preg_2 %>% select(datetime, count)

vroom_write(x=bike_predictions_preg_2, file="bike_predictions_preg_2.csv", delim=",")


lin_model <- linear_reg() %>%
  set_engine("lm")


preg_model <- poisson_reg(penalty = .25, mixture = .115) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model) %>%
  fit(data=log_train_set)

bike_predictions_preg <- predict(preg_wf, new_data = bike_test) %>%
  mutate(.pred = exp(.pred))

bike_predictions_preg$count <- bike_predictions_preg$.pred
bike_predictions_preg$datetime <- as.character(format(bike_test$datetime))
bike_predictions_preg <- bike_predictions_preg %>% select(datetime, count)

vroom_write(x=bike_predictions_preg, file="bike_predictions_preg.csv", delim=",")

preg_model_2 <- linear_reg(penalty = tune(),
                           mixture = tune()) %>%
  set_engine("glmnet")

preg_wf_2 <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model_2)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(log_train_set, v = 5, repeats = 1)

CV_results <- preg_wf_2 %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data =., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- preg_wf_2 %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_train_set)

bike_predictions_preg_2 <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))



vroom_write(x=bike_predictions_preg_2, file="bike_predictions_preg_2.csv", delim=",")

# REGRESSION TREE
my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

preg_wf_tree <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = (5))

folds <- vfold_cv(log_train_set, v = 5, repeats = 1)

CV_results <- preg_wf_tree %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel)

collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data =., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- preg_wf_tree %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_train_set)

bike_predictions_tree <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))



vroom_write(x=bike_predictions_tree, file="bike_predictions_preg_tree.csv", delim=",")

# RANDOM FORESTS
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

wf_rf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = (5))

folds <- vfold_cv(log_train_set, v = 5, repeats = 1)

CV_results <- wf_rf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq))

collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(data =., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

bestTune <- CV_results %>%
  select_best("rmse")

final_wf <- wf_rf %>%
  finalize_workflow(bestTune) %>%
  fit(data = log_train_set)

bike_predictions_rf <- final_wf %>%
  predict(new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))


vroom_write(x=bike_predictions_rf, file="bike_predictions_preg_rf.csv", delim=",")

# STACKED MODEL
bike_train <- vroom("./train.csv") %>%
  select(-casual, -registered) %>%
  mutate(count=log(count))


folds <- vfold_cv(bike_train, v = 5)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5)


preg_models <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=preg_tuning_grid,
            metrics=metric_set(rmse),
            control = untunedModel)

my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)
lin_reg_model <- fit_resamples(
  bike_workflow,
  resamples = folds,
  metrics = metric_set(rmse),
  control = tunedModel)


rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")


wf_rf <- workflow() %>%
  add_recipe(my_recipe2) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = (5))

rf_folds <- wf_rf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse),
            control=untunedModel)

bike_stack <- stacks() %>%
  add_candidates(preg_models) %>%
  add_candidates(lin_reg_model) %>%
  add_candidates(rf_folds)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

as_tibble(bike_stack)


stacked_predictions <- predict(fitted_bike_stack, new_data = bike_test) %>%
  mutate(.pred = exp(.pred)) %>%
  bind_cols(., bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=stacked_predictions, file="stacked_predictions.csv", delim=",")
