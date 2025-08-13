# viz_brainstorming.R
# Making some basic visualizations 
# I'm shopping for ideas

library(tidyverse)
library(paletteer)

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
  ggplot(aes(x = AQI, y = mask_usage_rate)) + geom_point() + 
  facet_wrap(~region)
