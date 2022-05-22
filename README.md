# Highway to the Danger Zone 𝅗𝅥 𝅗𝅥 𝅗𝅥 𝅗𝅥 𝅘𝅥 ♫ 𝅘𝅥
In this project, we develop a logistic regression model, dubbed the Highway model, to analytically identify the situations in which defensive play breaks down and in what situations does it successfully prevent shots by predicting the proability of a game state being a dangerous situation  (i.e. the probability of a high-danger unblocked shot by the power play within the three passes following the configuration). We additionally build an actionable tool (https://highway-to-the-danger-zone.netlify.app/ | Github: https://github.com/nguyenank/bdc22-mst-website) that coaches and analysts can use to apply to their own teams and strategies to both minimize and maximize high-danger shot attempts. Read the full writeup in the paper [Highway to the Danger Zone]([https://github.com/nguyenank/bdc22-mst/blob/main/Highway to the Danger Zone.pdf](https://github.com/nguyenank/bdc22-mst/blob/main/Highway%20to%20the%20Danger%20Zone.pdf "Highway to the Danger Zone.pdf").

The final merged and cleaned dataset used is [all_powerplays_4-23-22_cleaned_trimmed.csv](https://github.com/nguyenank/bdc22-mst/blob/main/all_powerplays_4-23-22_cleaned_trimmed.csv "all_powerplays_4-23-22_cleaned_trimmed.csv"). The 'pipeline' for running this project was the follwing:

1. Run [bdc_merge_example.ipynb](https://github.com/nguyenank/bdc22-mst/blob/main/bdc_merge_example.ipynb "bdc_merge_example.ipynb") to merge the data.
2. Manually clean and calculate the distance to attacking net according the processes described in the paper.
3. Run [modelling_and_plotting.py](https://github.com/nguyenank/bdc22-mst/blob/main/modelling_and_plotting.py "modelling_and_plotting.py").

#### Other notes 
- [hockey_mst.py](https://github.com/nguyenank/bdc22-mst/blob/main/hockey_mst.py "hockey_mst.py") is used to outline functions for handing the minimum spanning tree calculations. 
- [feature_values.csv](https://github.com/nguyenank/bdc22-mst/blob/main/feature_values.csv "feature_values.csv") is the human readable list of coefficients used in the logistic regression model
- [feature_dict.json](https://github.com/nguyenank/bdc22-mst/blob/main/feature_dict.json "feature_dict.json") is the machine readable JSON list of coefficients used in the logistic regression model 
- [tool_test_data](https://github.com/nguyenank/bdc22-mst/tree/main/tool_test_data "tool_test_data") folder contains the plots and variable values used to calibrate the webtool
- [graphic_output](https://github.com/nguyenank/bdc22-mst/tree/main/graphic_output "graphic_output") contains model evaluation charts.
- [high_danger_states](https://github.com/nguyenank/bdc22-mst/tree/main/high_danger_states "high_danger_states") and [low_danger_states](https://github.com/nguyenank/bdc22-mst/tree/main/low_danger_states "low_danger_states") contain examples of game states from the dataset that had extremely either high or low probabilities of being a dangerous situation.
- [Old Data](https://github.com/nguyenank/bdc22-mst/tree/main/Old%20Data "Old Data") contains previously used testing data 
