# Reference:
# - https://github.com/trestletech/plumber
# - https://github.com/Knowru/plumber_example/blob/master/deploy_ml_credit_model.R

library(rpart)
library(jsonlite)
load("decision_tree_for_german_credit_data.RData")

#* @get /
health.check <- function() {
  return("ok")
}

#* @post /predict
predict.default.rate <- function(input) {
  if(typeof(input)=="character") {
    in_data = fromJSON(input)
  } else {
    in_data = input
  }

  df = data.frame(in_data)
  # manintaining types is very important, taking a shortcut below as this is just an example. But, this is where
  # explicitly mentioning types outlasts the pain one may encounter later
  df$Duration.in.month <- as.numeric(as.character(df$Duration.in.month))
  prediction <- predict(decision.tree, df)

  # For simplification just returning a single prediction
  return(list(probability=prediction[,1]))
}
