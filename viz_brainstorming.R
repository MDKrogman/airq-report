# viz_brainstorming.R
# Making some basic visualizations 
# I'm shopping for ideas

library(tidyverse)
library(paletteer)
library(corrplot)
library(patchwork)

airq <- read_csv('air_quality_health_dataset.csv')

airq <- airq %>% 
  separate(date, into = c('year', 'month', 'day'), sep = '-') # Use this

airq1 <- airq %>% 
    group_by(month, region) %>% 
    summarize(n_lockdowns = sum(lockdown_status))

airq1 %>% 
  ggplot(aes(x = month, y = n_lockdowns)) +
  geom_point(aes(color = region))

# putting a pin in this one for a rq

airq2 <- airq %>% 
  group_by(month, region) %>% 
  summarize(n_hospit = sum(hospital_visits) + sum(emergency_visits))

airq2 %>% 
  ggplot(aes(x = month, y = n_hospit)) +
  geom_point(color = 'navy') +
  geom_line(aes(group = region), color = 'navy') + 
  facet_grid(~region)

# There's something here but this leads me to a new idea

airq3 <- airq %>% 
  mutate(n_hospit = hospital_visits + emergency_visits,
         .before = 18)

airq3 %>% 
  ggplot(aes(x = mask_usage_rate, y = n_hospit)) + geom_point() + 
  facet_wrap(~region)

# It seems there is very little information to be gained from this
# there's basically no trend at all

# Making a correlation plot now

airq_cor <- cor(airq[5:28])
corrplot(airq_cor, method = 'shade')  # variables have none or very minimal correlation

# New idea after playing with models

airq4 <- airq %>% 
  pivot_longer(cols = 6:11,
               names_to = 'pollutant',
               values_to = 'pollutant_level'
              ) %>% 
  relocate(pollutant, pollutant_level, .after = AQI)

airq4$pollutant <- factor(airq4$pollutant, levels = c('CO', 'SO2', 'O3', 'NO2', 'PM2.5', 'PM10'))

p1 <- ggplot(airq4, aes(x = pollutant_level, y = lockdown_status, color = pollutant)) +
  geom_jitter(height = .4, alpha = .4) + 
  scale_y_continuous(breaks = c(0, 1)) +
  theme_minimal() +
  scale_colour_paletteer_d("MetBrewer::Juarez") +
  theme(
    legend.position = 'none'
  ) +
  labs(
    x = '', y = 'Lockdown Status',
    color = 'Pollutant'
  )

p2 <- ggplot(airq4, aes(x = pollutant_level, y = school_closures, color = pollutant)) +
  geom_jitter(height = .4, alpha = .4) + 
  scale_y_continuous(breaks = c(0, 1)) +
  theme_minimal() +
  scale_colour_paletteer_d("MetBrewer::Juarez") +
  theme(
    legend.position = 'bottom'
  ) +
  labs(
    x = 'Pollutant Level', y = 'School Closure Status',
    color = 'Pollutant'
  )

p1 / p2

# detailing these more because I think these will be useful visualizations
# pay attention to when the pollutant tends to be lower on days when the lockdowns/school closures occur
# also note that the number of dots has effectively been multiplied by 6 since we pivoted the data

airq4 %>% 
  filter(pollutant == 'PM2.5') %>% 
  ggplot(aes(x = pollutant_level, y = AQI, color = region)) + geom_point(alpha = .5) +
  facet_wrap(~month) + 
  scale_colour_paletteer_d("MetBrewer::Juarez")

airq4 %>% 
  filter(pollutant == 'PM2.5') %>% 
  ggplot(aes(x = pollutant_level, y = respiratory_admissions, color = region)) + geom_point(alpha = .5) +
  facet_wrap(~month) + 
  scale_colour_paletteer_d("MetBrewer::Juarez") +
  labs(
    x = 'PM2.5 Level', y = 'Respiratory Hospital Admissions',
    color = 'Region'
  )

# May be useful in conjunction w/ a time series
