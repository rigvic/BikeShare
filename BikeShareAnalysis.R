library(tidyverse)
library(tidymodels)
library(vroom)

bike_train <- vroom("./train.csv")
bike_test <- vroom("./test.csv")

# cleaning
bike_train <- bike_train %>%
  select(-casual, -registered) %>%
  mutate(weather = ifelse(weather == 4, 3, weather))

# feature engineering
my_recipe <- recipe(count~., data = bike_train) %>%
  step_num2factor(season, levels = c("Spring", "Summer", "Fall", "Winter")) %>%
  step_num2factor(weather, levels = c("Clear", "Mist", "Rain")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime)
  

my_recipe <- prep(my_recipe)
bake(my_recipe, new_data = bike_train)
baked_recipe <- bake(my_recipe, new_data = bike_test)







