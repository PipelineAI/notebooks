### Deploying a model via flask

Ensure flask, numpy, and scikit-learn are available in your environment.

To run, execute api/model.py, which will host the model as a service. The service should be available on localhost, so the model endpoint would be:

-http://localhost:5000/predict
or
-http://127.0.0.1:5000/predict


```
def input_formatter(data):
    return {'input':data.tolist()}
    
def output_formatter(response, key='predictions'):
    print
    return np.array(response.json()[key])

uri = 'http://datsci.dev:5000/predict'

dep_model = DeployedModel(uri, 
                         input_formatter,
                         output_formatter,
                         target_names=['Housing Prices'], 
                         examples=X_boston[:1000])
                         
```
