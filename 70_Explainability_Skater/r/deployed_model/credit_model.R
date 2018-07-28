# For details related to this German credit data set, please refer https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
col.names <- c(
'Status of existing checking account', 'Duration in month', 'Credit history'
, 'Purpose', 'Credit amount', 'Savings account/bonds'
, 'Employment years', 'Installment rate in percentage of disposable income'
, 'Personal status and sex', 'Other debtors / guarantors', 'Present residence since'
, 'Property', 'Age in years', 'Other installment plans', 'Housing', 'Number of existing credits at this bank'
, 'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign worker', 'Status'
)
# Get the data
data <- read.csv(
url
, header=FALSE
, sep=' '
, col.names=col.names
)

library(rpart)
# Build a tree
decision.tree <- rpart(
Status ~ Status.of.existing.checking.account + Duration.in.month + Credit.history + Savings.account.bonds
, method="class"
, data=data
)

install.packages("rpart.plot", repos='http://cran.us.r-project.org')
library(rpart.plot)
# Visualize the tree
# 1 is good, 2 is bad
prp(
decision.tree
, extra=1
, varlen=0
, faclen=0
, main="Decision Tree for German Credit Data"
)

# Save the tree for deployment
save(decision.tree, file='decision_tree_for_german_credit_data.RData')