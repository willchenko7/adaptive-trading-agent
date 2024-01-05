# adaptive-trading-agent

A challenging problem in computational intelligence lies in effectively designing models that have no predefined output label. 

The market is one of these situations. It has a clear goal (make line go up) and clear input data (previous price/volume data), but there is no clear strategy that is the best.

Most (naive) trading strategies simplify their algorithms by relying on a predict-the-next-price structure. In this situation, there is a clear label (the actual next price) that the model can be trained to predict. However, these strategies are incapable of beating out a simple linear regression, and it should be obvious that this strategy will have mediocre results.

More generally, any method that is predefined by the human can easily be transferred to a computational agent by labelling the data in the way the human wants it to be. Essentially telling the agent, "trade like this." But this clearly is limited to the imagination and intellect of its human.

So then the question becomes, how can an artificial system come up with its own trading system where it is unclear to the user what the best strategy is? In this repo we will explore different strategies and attempt to answer this question.

Hopefully, the lessons learned here can be translated to other complex systems where the ideal strategy is unknown.

## Methods
The method here is to use a handrolled evolutionary algorithm to adjust the weights of a neural network with attention to optimize for trading success.

The trading simulation works by passing normalized historical price data for each coin through the forward pass of the optimized model. The output will be a continuous value between 0 and 1 for each coin. Then, the model will use pre-defined logic to hold current coin or trade for a new one. If the output of the current coin is less than the sell_mark (sell_mark=0.1), then it will trade for the coin with the highest output at that point. Each transaction costs 2.7% to simulate gas fees. This process repeats for the simulation length.

In this way, there is no predefined strategy for trading and the model can learn however it wants to make money.

Note: the model can also be trained for more than one objectives by passing multiple simulation start times to the eveolutionary algorithm. The fitness will be the average of each sim result.

## Usage
First, you will need to clone this repo, cd to the root of this project, create a venv, and download all necessary requirements.

You can run the evolutionary algorithm by running the following command in bash:
```
python3 src/ea.py
```

You can arrive at the "best_model" by testing all of your models on unseen data and saving the best model with:
```
python3 src/test_models.py
```

You can test the best model on the most recent data with:
```
python3 src/most_recent_sim.py
```
This will also let the user know what some potential next steps could be at the current moment.

You can update the data to the current time with:
```
python3 src/update_all_data.py
```