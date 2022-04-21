require(tidyverse)

xbreaks <- seq(-100.01, 100.01, length.out = 11)
#goes from -100.01 to 100.01 to properly account values at 100 or -100
ybreaks <- seq(-42.6, 42.6, length.out = 8)
#goes from -42.6 to 42.6 to properly account values at 42.5 or -42.5
id <- matrix(seq_len(length(xbreaks) * length(ybreaks)),
             length(xbreaks), length(ybreaks))