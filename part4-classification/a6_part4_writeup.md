# Part 4 - Classification Writeup

After completing `a6_part4.py` answer the following questions

## Questions to answer

1. Comment out the StandardScaler and re-run your test. How accurate is the model? Why is that?
The model with the standardscaler is mostly accurate. Out of 80 tests it failed 16 of them which means it is partially accurate but not enough for consistent use.

2. How accurate is the model with the StandardScaler? Is this model accurate enough for the given use case? Explain.
The model's accuracy changed slightly with the StandardScaler. Out of 80 tests it failed 14 of them. However this probably isn't due to the scaler but is due to the differing testing and training data. The model is still not accurate enough for consistent use.
3. Looking at the predicted and actual results, how did the model do? Was there a pattern to the inputs that the model was incorrect about?
The model, generally, did well. Though there wasn't perfect accuray the model was mdefinately much better than making complete guesses.
4. Would a 34 year old Female who makes 56000 a year buy an SUV according to the model? Remember to scale the data before running it through the model.
Not according to this model. The model said the kind of person who would do this is a male.
