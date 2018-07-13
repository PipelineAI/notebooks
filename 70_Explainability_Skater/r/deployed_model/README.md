* Follow the instructions in the below mentioned reference to create a
credit card model. 
Reference: 
    - https://github.com/Knowru/plumber_example/blob/master/create_ml_credit_model.R
* Once the model is build, invoke the following steps to start the plumber
daemon
    - cd `Skater/examples/r`
    - launch an R kernel from the terminal 
    - setwd("deployed_model/")
    - install.packages('plumber')
    - library(plumber)
    - source("credit_model.R")
    - r_example <- plumb("deploy_ml_credit_model.R")
    - r_example$run(port=8000)
    - **Note:** Its possible that if you have an existing process running on 
      port:8000 the plumber daemon may not come up. Try using 
      ``` netstat -an | grep 8000 | grep -i listen```. If nothing is returned
      all is good.
* Use the following link to visit the url http://localhost:8000/predict
* Would recommend to use to postman to query the endpoint and validate
  if things are fine. 
* Possible input 
  ```
  {"Status.of.existing.checking.account": "A14", 
  "Duration.in.month": 12, 
  "Credit.history": "A34", 
  "Savings.account.bonds": "A65"}
  ```
  expected output: 
  ```
  {
    "default.probability": 0.1313
  }
  ```