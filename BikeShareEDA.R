## ttest

library(tidyverse)
library(vroom)
library(patchwork)
bike <- vroom("./train.csv")

dplyr::glimpse(bike)
plot1 <- DataExplorer::plot_intro(bike)
plot2 <- DataExplorer::plot_correlation(bike)
plot3 <- ggplot(data=bike, aes(x=temp, y=count)) + geom_point() +geom_smooth(se=FALSE)
plot4 <- DataExplorer::plot_missing(bike)
GGally::ggpairs(bike)
(plot1 + plot2) / (plot4 + plot3)
