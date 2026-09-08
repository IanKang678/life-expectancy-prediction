# Predicting Life Expectancy from Public Health Indicators

This was a paired project with Mark Jayaraman in the Skew The Script Data Science & AI Bootcamp. This version reflects some independent revisions I did.

## Question
Can AI and Data Science be used to predict a country's life expectancy? Can health and economic indicators help predict it, and what factors affect a country's life expectancy the most?

## Data
The data was from the World Health Organization Global Health Observatory via Kaggle. The data contained 193 countries, spanned the years 2000 to 2015, and was about 2864 rows with 20 predictors. Each row is one country in one year. The file is committed to `data/`, so the notebook reads it locally instead of fetching it from a URL. Two packages are required:

```r
install.packages(c("coursekata", "neuralnet"))
```

## Method
Starting with a mean RMSE baseline of 9.78 years, the dataset was reorganized into an 80/20 training/testing split. Regression models using a first-pass set of predictors were tried. To create a better regression model, a correlation table was used to find variables which were independent and had an effect on life expectancy. Using the findings from the correlation table, 10 polynomial regression models were created using degrees of 1 to 10 on GDP per capita, but a logarithmic shape in the GDP per capita vs life expectancy scatterplot led to the creation of a logarithmic regression model. Finally, two neural network architectures with different layers and nodes were tried. 5 repetitions were done on each model.

## Findings
**The neural networks in the original project never trained.** Both returned an RMSE of 9.78, equivalent to each other to 8 decimals, with prediction standard deviation at 0. Every country got a life expectancy of 68.98, the training mean. The reason for this was because the predictors were scaled to between 0 and 1, but the target was not, sitting at 60 to 80. The networks could not reach the output range and settled on an intercept. The original write-up read the identical RMSEs as evidence that architecture did not matter, when they were actually evidence that neither model had learned anything.

After scaling the target, both networks trained.

| Model | Test RMSE (years) |
|---|---|
| Mean baseline | 9.78 |
| Regression, first-pass predictors | 2.91 |
| Regression, log GDP | 1.93 |
| Neural network, 8-4-2 | 1.68 |
| Neural network, 32-16-8-4-2 | 1.51 |

The network results are means across 5 training runs because a single run is not reproducible. The three layer network ranged from 1.644 to 1.694, while the 5 layer network ranged from 1.226 to 1.667. The deeper network is better on average, but far less consistent, its spread is about 0.44 years against 0.05 for the shallower one.

Each polynomial regression model, from degrees 1 to 10, ranged from an RMSE of 1.952 to 1.757, reaching a minimum at degree 8 and rising again at 9 and 10. These were scored on a validation set held out from the training data, so they are not comparable to the test RMSEs above. The differences were small between degrees, and the degree 10 polynomial when plotted showed signs of overfitting the data as there were curves in places with no data. This leads to the logarithmic regression model which had an RMSE of 1.926.

When evaluated on the 2015 dataset, the best network got an RMSE of 1.47, while the chosen regression model, logarithmic, got an RMSE of 2.02.

One interesting finding was that the BMI coefficient turned negative when under control. Without control, an increase in it seems to increase life expectancy, but with control, a unit increase is associated with a life expectancy decrease of about 0.0764 years. It also had a p value of 0.0015 showing that it was significant.

## Validation and leakage
There were three leaks in the project, two of which were fixed.

The original part 5 evaluation was on the entire 2015 dataset, 80% of which was training data. The neural model had used that data to train, causing it to produce results which were more accurate than it should have been. This was fixed by filtering from only the test set, data the network had not seen before.

The polynomial degree had also been chosen by test set RMSE, which lets the test set influence a modeling decision. This now uses a validation split held out from the training data. Similarly, the neural network scaling divided by maxima calculated over the full dataset including test rows, and now uses training maxima only.

The remaining leak is that the data was split randomly, meaning that for example, Germany 2004 could be training while Germany 2005 could be test. Those two datapoints are likely close together, so the model can interpolate between adjacent years rather than genuinely predict, and the reported RMSEs are likely optimistic as a result. Fixing this needs a time based split, which would change every number in the project.

## Limitations
There were three main limitations. 

First of all, the random split ignores the time structure, and since life expectancy rose over 2000–2015 while the model learned an average relationship across the whole period, it under-predicted the 2015 life expectancy by 0.41 years.

Another limitation was the worst predictions happening in unstable states. For example, the model under-predicted Syria by 3.7 years, with Thailand, Kiribati, Malawi and Somalia next. Due to conflict, many predictors were low, causing the model to drop the predicted life expectancy by a lot.

Finally, several network repetitions hit the step limit without converging. `neuralnet` stops either when the gradient falls below the threshold, meaning it finished, or when it runs out of steps, meaning it did not. Some runs stopped the second way, so those models were still improving when training ended and the architecture comparison is partly between partially trained networks.

## Future work
A time based split, training on 2000–2014 and testing on 2015, would remove the interpolation problem and test the model on forecasting rather than filling in gaps.

The two architectures were compared on the test set, the same problem the polynomial sweep was fixed for. Choosing the architecture on a validation split would make the final test RMSE a clean estimate instead of the best of several attempts.

Running the networks to convergence would also make the comparison fair. Raising the step limit and lowering the threshold would let every run finish, which should also reduce the run to run variation seen in the deeper network.

## Repo contents
- `notebooks/analysis.ipynb` — the full analysis, from EDA through evaluation
- `data/public_health_data.csv` — the WHO dataset, committed so the notebook runs standalone
- `plots/` — exported figures