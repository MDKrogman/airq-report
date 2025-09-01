# prelim_models.R
# My first chunk of work in connection to building models
# Might do some machine learning work here, not sure yet, though.

library(tidyverse)
library(caret)
library(tidymodels)
library(vip)


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

# rec3 was the best one but I still found the rmse kinda big so we're going to try and get something better

rec5 <- rec3 %>% 
  step_nzv(all_numeric_predictors())  # I figured this unnecessary earlier but it really can't hurt

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
  preproc = list(rec3, rec5),
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
                              select_best(wf_best_tuned, metric = 'rmse')) # In case everything needs to be rerun: rec3_rand_forest1
wf_final_fit <- last_fit(wf_final, index)                                  

final_model <- fit(wf_final, data = airq_hospit)

predictions <- predict(final_model, new_data = airq_hospit)
predictions$.pred <- ceiling(predictions$.pred) # rounding up because you can't have part of a bed of course
results <- bind_cols(airq_hospit, predictions)

# so now .pred gives us an approximation of how many people will be in hospitals throughout the entire day
# so, they'll know how to allocate their resources appropriately.
# the only thing that could make this more specific would be to have hour-by-hour data





# Going to make another machine learning model, for predicting lockdowns

airq_lockdown <- airq %>% 
  mutate(lockdown_status = as.factor(lockdown_status)) %>% 
  select(-c(year, day))   # removing these because I fail to see how they would be any more than incidental
                          # same reason as in the last model too

index2 <- initial_split(airq_lockdown, prop = .8)
train2 <- training(index2)
test2 <- testing(index2)
cv_folds2 <- vfold_cv(data = train2, v = 10)

glm_rec1 <- recipe(lockdown_status ~ ., data = train2) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors())

glm_rec2 <- glm_rec1 %>% 
  step_zv(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_interact(terms = ~ (PM2.5 + PM10 + NO2 + SO2 + CO + O3)^2)

glm_rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  mode = 'classification',
  engine = 'ranger'
)

glm_log_spec <- logistic_reg(
  mode = 'classification',
  engine = 'glm'
)

glm_log_spec2 <- logistic_reg(
  penalty = tune(),
  mode = 'classification',
  engine = 'glmnet'
)

glm_knn_spec <- nearest_neighbor(
  mode = 'classification',
  engine = 'kknn',
  neighbors = tune()
)

glm_wf_set <- workflow_set(
  preproc = list(glm_rec1, glm_rec2),
  models = list(glm_rf_spec, glm_log_spec, glm_log_spec2, glm_knn_spec),
  cross = TRUE
)

glm_fit_workflows <- glm_wf_set %>% 
  workflow_map(
    fn = 'tune_grid',
    grid = 10,
    resamples = cv_folds2,
    verbose = TRUE
  )

autoplot(glm_fit_workflows)

glm_best_wf_id <- rank_results(glm_fit_workflows, rank_metric = 'accuracy',
                           select_best = TRUE) %>% 
  slice_head(n = 1) %>% 
  pull(wflow_id)

glm_wf_best <- extract_workflow(glm_fit_workflows,   # Gotta pull out the best workflow
                            id = glm_best_wf_id)

glm_wf_best_tuned <- glm_fit_workflows[glm_fit_workflows$wflow_id == glm_best_wf_id,
                               'result'][[1]][[1]]

collect_metrics(glm_wf_best_tuned) %>%
  filter(.metric == 'accuracy') %>%     
  arrange(desc(mean))

glm_wf_final <- finalize_workflow(glm_wf_best,
                              select_best(glm_wf_best_tuned, metric = 'accuracy')) # gives a pretty definitive tune settings
glm_wf_final_fit <- last_fit(glm_wf_final, index2)                                  

final_glm <- fit(glm_wf_final, data = airq_lockdown)

predictions_glm <- predict(final_glm, new_data = airq_lockdown)
results_glm <- bind_cols(airq_lockdown, predictions_glm)

results_glm %>% 
  select(c(lockdown_status, .pred_class)) %>% 
  View()


# some more non-ML stuff here

lockdown_glm <- glm(lockdown_status ~ (PM2.5 + PM10 + NO2 + SO2 + CO + O3)^2, data = airq_lockdown, family = 'binomial')
summary(lockdown_glm)

lockdown_glm2 <- glm(lockdown_status ~ (PM2.5 + SO2 + CO)^2, data = airq_lockdown, family = 'binomial')
summary(lockdown_glm2)

# The most significant term here is the PM2.5:SO2 interaction term. According to a quick google search
# wildfires/agricultural burning release SO2 into the atmosphere (!)
# https://cpo.noaa.gov/quantifying-sulfur-dioxide-emissions-and-understanding-air-quality-impacts-from-fire-activity/

# making a similar model for hospital beds

hospit_glm <- glm(n_hospit ~ (PM2.5 + PM10 + NO2 + SO2 + CO + O3)^2, data = airq_hospit, family = 'poisson')
summary(hospit_glm)

hospit_glm2 <- glm(n_hospit ~ (PM10 + CO + SO2)^2, data = airq_hospit, family = 'poisson')
summary(hospit_glm2)

# with both of these models we are able to get some evidence for effect on lockdown status/number of beds on a day
# but with such interactions we lose some ability to interpret.
# note that I also made a choice to just include the pollutants.
# that's a benefit of randomforests like what my ML models use: they can handle those better.


p3 <- ggplot(results, aes(results, x = n_hospit, y = .pred)) +
        geom_point(alpha = 0.5, color = 'steelblue') +
        geom_abline(slope = 1, intercept = 0, linetype = 'dashed') +
        theme_minimal() +
        labs(
          x = 'Actual Hospital Demand', y = 'Predicted Hospital Demand',
          title = 'Model Predictions vs. Actual Values'
        ) 

# Can't run this just yet, need to wait for sourcing to stop running