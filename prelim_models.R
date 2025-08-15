# prelim_models.R
# My first chunk of work in connection to building models
# Might do some machine learning work here, not sure yet, though.

library(tidyverse)
library(caret)
library(tidymodels)


airq <- read_csv('air_quality_health_dataset.csv') %>% 
  separate(date, into = c('year', 'month', 'day'), sep = '-')


# Starting with some models that may allow us to work towards a ML model that 
# predicts the need for hospital beds in the area

airq_hospit <- airq %>% 
  mutate(n_hospit = hospital_visits + emergency_visits + respiratory_admissions) %>% 
  select(-c(year, day, hospital_visits, emergency_visits, respiratory_admissions))

hospit_model <- lm(n_hospit ~ ., data = airq_hospit)
summary(hospit_model)

ihospit_model <- lm(n_hospit ~ . + PM2.5 * PM10 * NO2 * SO2 * CO * O3, data = airq_hospit)
summary(ihospit_model) # so far this is the most effective model we have

# should really cut the variables down a bit

hospit_model2 <- lm(n_hospit ~ month + region +AQI + temperature + humidity + wind_speed + 
                      precipitation + mobility_index + school_closures + 
                      public_transport_usage + mask_usage_rate + lockdown_status + 
                      industrial_activity + vehicle_count + construction_activity + 
                      population_density + green_cover_percentage, data = airq_hospit)
summary(hospit_model2)

# so we still can't really discern anything from this

ihospit_model2 <- lm(n_hospit ~ month  + region  + PM2.5  + temperature  + humidity  + wind_speed  + 
                      precipitation  + mobility_index  + school_closures  + 
                      public_transport_usage  + mask_usage_rate  + lockdown_status  + 
                      industrial_activity  + vehicle_count  + construction_activity  + 
                      population_density + green_cover_percentage + mobility_index * . + month * region, data = airq_hospit)
summary(ihospit_model2)

feature_importance <- varImp(ihospit_model2)
feature_importance %>% 
  arrange(desc(Overall))

step_process <- MASS::stepAIC(ihospit_model2, direction = 'both')

AIChospit_model <- lm(n_hospit ~ month + region + PM2.5 + temperature + humidity + 
                        wind_speed + precipitation + mobility_index + school_closures + 
                        public_transport_usage + mask_usage_rate + lockdown_status + 
                        industrial_activity + vehicle_count + construction_activity + 
                        population_density + green_cover_percentage + mobility_index *
                        (month + region + AQI + PM2.5 + PM10 + NO2 + SO2 + CO + O3 + 
                           temperature + humidity + wind_speed + precipitation +
                           mobility_index + school_closures + public_transport_usage +
                           mask_usage_rate + lockdown_status + industrial_activity + 
                           vehicle_count + construction_activity + population_density + 
                           green_cover_percentage) +  month * region, data = airq_hospit)
summary(AIChospit_model)

# going to try a similar model now that takes exclusively things that would direct towards hospitalizations
# plus a few other things

harmful_hospit_model <- lm(n_hospit ~ AQI + PM2.5 + PM10  + SO2 + CO + 
                            construction_activity, data = airq_hospit)
summary(harmful_hospit_model)

# note at end of work session: ihospit_model is our best/most intriguing so far,
# but should also look more into that mobility_index variable

# I will actually do some ML here
# A model to predict the number of hospital visits, regardless of emergency or no
# would be helpful for determining staff/other resource needs

set.seed(449)

# skipping zv/correlation steps because we already know their unimportance

index <- initial_split(airq_hospit, prop = .8)
train <- training(index)
test <- testing(index)
cv_folds <- vfold_cv(data = train, v = 10)

rec1 <- recipe(n_hospit ~ ., data = train) %>% 
  step_dummy(region, month)

rec2 <- rec1 %>% 
  step_normalize(all_numeric_predictors()) 

rec3 <- rec2 %>% 
  step_interact(terms = ~ (PM2.5 + PM10 + NO2 + SO2 + CO + O3)^2) # same interactions as in ihospit_model

rec4 <- rec1 %>% 
  step_interact(terms = ~ (PM2.5 + PM10 + NO2 + SO2 + CO + O3)^2) # curious to see if normalization has a huge effect

rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  mode = 'regression',
  engine = 'randomForest'
)

rf_spec2 <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  mode = 'regression',
  engine = 'ranger'
)

knn_spec <- nearest_neighbor(
  mode = 'regression',
  engine = 'kknn',
  neighbors = tune()
)

wf_set <- workflow_set(
  preproc = list(rec1, rec2, rec3, rec4),
  models = list(rf_spec, rf_spec2, knn_spec),
  cross = TRUE
)

fit_workflows <- wf_set %>% 
  workflow_map(
    fn = 'tune_grid',
    grid = 10,
    resamples = cv_folds,
    verbose = TRUE
  )

best_wf_id <- rank_results(fit_workflows, rank_metric = 'rmse',
                           select_best = TRUE) %>% 
  slice_head(n = 1) %>% 
  pull(wflow_id)

wf_best <- extract_workflow(fit_workflows,   # Gotta pull out the best workflow
                            id = best_wf_id)

wf_best_tuned <- fit_workflows[fit_workflows$wflow_id == best_wf_id,
                               'result'][[1]][[1]]

collect_metrics(wf_best_tuned) %>%
  filter(.metric == 'rmse') %>%     
  arrange(mean)

wf_final <- finalize_workflow(wf_best,
                              select_best(wf_best_tuned, metric = 'rmse')) # In case everything needs to be rerun: rec3_rand_forest2
wf_final_fit <- last_fit(wf_final, index)                                  # mtry = 1, min_n = 14, on ranger

final_model <- fit(wf_final, data = airq_hospit)

predictions <- predict(final_model, new_data = airq_hospit)
predictions$.pred <- ceiling(predictions$.pred) # rounding up because you can't have part of a bed of course
results <- bind_cols(airq_hospit, predictions)

results %>%                             # Looking at differences
  group_by(month) %>%                   
  select(c(month, n_hospit, .pred)) %>% 
  View()
