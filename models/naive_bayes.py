import numpy as np
from models.base_model import BaseModel

# for classification 
#we are making a mathematical bet that for every class, 
# each feature follows a Normal (Gaussian) Distribution.
class GaussianNaiveBayes(BaseModel):
    def __init__(self, var_smoothing=1e-9):
        
        #Gaussian Naive Bayes (NB) Classification.
        #var_smoothing : float
        # Portion of the largest variance of all features that is added to 
        # variances for calculation stability.
        self.var_smoothing = var_smoothing
        self.classes = None
        #the Prior for a class is simply the percentage of the training data that belongs to that class.
        self.parameters = [] # List of dicts containing mean, var, and prior per class


    # train the model by calculating mean ,variance and priors for each class     
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.parameters = []
        # here we group rows refering to the class of each one 
        for _, c in enumerate(self.classes):
            # Filter X for only this class
            X_c = X[y == c]
            
            # Calculate mean and variance for each feature in this class
            # Adding var_smoothing to variance to avoid division by zero
            stats = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0) + self.var_smoothing,
                "prior": X_c.shape[0] / X.shape[0]
            }
            self.parameters.append(stats)
    #Calculate Gaussian likelihood P(x_i | class)        
    def _calculate_likelihood(self, class_idx, x):
        mean = self.parameters[class_idx]["mean"]
        var = self.parameters[class_idx]["var"]
        # Gaussian PDF formula
        # P(x|mean, var) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def _predict_single(self, x):
        posteriors = []
        for i, _ in enumerate(self.classes):
            # We use log-probabilities to prevent numerical underflow
            #P(C)
            prior = np.log(self.parameters[i]["prior"])
            #P(x/C)
            likelihood = np.sum(np.log(self._calculate_likelihood(i, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        #(log-probabilities are usually negative. The number closer to zero is the "higher" probability).
        #np.argmax finds the index of the largest number

        #Since $P(x)$ is the exact same number for both calculations, it doesn't change the "ranking."
        return self.classes[np.argmax(posteriors)]
    

    #Predict labels using the Maximum A Posteriori (MAP) rule.
    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    