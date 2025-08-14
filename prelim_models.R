# prelim_models.R
# My first chunk of work in connection to building models
# Might do some machine learning work here, not sure yet, though.

library(tidyverse)
library(caret)

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
