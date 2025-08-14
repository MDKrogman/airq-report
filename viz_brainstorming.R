# viz_brainstorming.R
# Making some basic visualizations 
# I'm shopping for ideas

library(tidyverse)
library(paletteer)
library(corrplot)

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


