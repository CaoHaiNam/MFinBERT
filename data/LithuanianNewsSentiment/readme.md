The analyzed dataset consisted of 10,375 texts, where 5780 texts were assigned to the
class positive (POS), neutral (NEU)—1997 texts, and 2598 texts were negative (NEG)

# Experiments
experimental investigation to find the best classifier for the analyzed
dataset, cross-validation and hyperparameters optimization with a grid search optimizer
was used. Each experiment was repeated five times and the average accuracy was evalu-
ated. For cross-validation, the commonly used 5-fold option was selected (80% training
dataset, 20% testing dataset); moreover, the so-called stratify distribution was used, which
helped to keep the same number of the class in each fold.