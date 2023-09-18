library(tidyverse)
library(tidymodels)
library(vroom)

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

library(poissonreg)

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

