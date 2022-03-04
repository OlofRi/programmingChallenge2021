# CODE WRITTEN BY: Olof Rickhammar, sources used are referenced in comments

# plotting, vectors, data management
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Model selection : https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # check for more like this one: https://www.youtube.com/watch?v=6dbrR-WymjI&ab_channel=DataSchool
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV # grid search for hyper parameters: https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8&ab_channel=DataSchool

# Feature selection : https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Different models
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # check other variants
from sklearn.neural_network import MLPClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture

from xgboost import XGBClassifier

# Scoring: around 80% accuracy or above on the evaluation data gives max score 18p.

# Using VS code with .csv extensions to analyze and clean data:
#   - Edit csv 0.5.6
#   - Excel Viewer 3.0.41

####### Data ##########
# Initial thoughts:
    # EvaluateOnMe.csv contained 10000 evaluation datapoints (1000-10999)
    # TrainOnMe.csv contained 1000 training datapoints (0-999) 
    # There were some 8 error lines with icelandic language between datapoints 257 and 258 (SINCE REMOVED)
    # There are some missing labels for datapoints. (How should this be handled?) (SINCE REMOVED)
    # There are some (unreasonably) large values for certain features. (How should this be handled?) (SINCE REMOVED)

# In-depth analysis TrainOnMe.csv:
    # File structure: index, y, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 (index is not represented by a str in header)
    # index = (0-999)
    # y = label = {Shoogee, Atsuto, Bob, Jorg}          # verify that no other labels exists. (VERIFIED)
    #       Missing labels: 173, 696, 992
    # x1 = real number, positive/negative, range approx:  -40 to +40. 
    #       Potential value errors: 734, 822 (VERY LARGE VALUES COMPARED TO RANGE)
    # x2 = negative real number, all values: -15.xxxxx. (No missing values...) (tight range, does not say that much? - PCA?) (FOUND CORRELATION OF 1 WITH x11)
    # x3 = real number, positive/negative, range approx: -1 to +1; +-0.xxxxx 
    # x4 = real number, positive/negative, range approx: -12 to +30
    # x5 = real number, positive/negative, range approx: -17 to +13
    # x6 = labeled values = {GMMs and Accordions, Bayesian Inference}   # verify that no other labels exists. (VERIFIED)
    #       Missing labels: 438, 606
    #       Misspelled labels: 189, 316 ("Bayesian Interference" instead of "Bayesian Inference") (FIXED SINCE)
    # x7 = real number, positive/negative, range approx: -30 to +30
    # x8 = real number, positive/negative, range approx: -50 to +60
    # x9 = negative real number, range approx: -70 to -80
    # x10 = positive real number, range approx: 2 to 11
    # x11 = negative real number, all values -19.xxxxx or -20.xxxxx (FOUND CORRELATION OF 1 WITH x2)
    # x12 = labeled values = Boolean = {True, False}
    #       Misspelled labels: 208, 714 -> "Flase" (FIXED SINCE to "False")

# After initial cleaning, the 7 datapoints: 173, 696, 992, 734, 822, 438, 606: have errors that can be corrected by guessing labels or changing values.
# Since they represent less than 1% of the data I will try to remove them first and maybe later use guessed values if I believe performance is affected.
#   (saved a copy in TrainOnMe copy.csv before deleting avove datapopints)
# https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4

######### Algorithm/Process: ########
# 0. Data cleaning
#   a. remove obvious wrong data (empty lines etc)
#   b. handle missing features and labels (fill with ex. average, randomize, or remove datapoint)
#   c. adjust unreasonable values (extreme variation?)
# 1. Process data from .csv file
#   a. import data from .csv file     # Is it possible to extract directly from csv file to numpy matrix?
#   b. read data and add to matrix / vector
#   c. Separate data into features X and labels Y
# 2. Feature selection (automatic)
#   a. Split data into training and test data: X-train, Y-train ; X-test, Y-test
# 3. Train ML-model on training data
# 4. Test model on test data
# 5. Evaluate performance and adjust parameters (at step 2-3)

# 6. Use finished best performing model to classify evaluation data
# 7. Write label classifications to .txt file
# 8. Double check all labels and format so that nothing is wrong
#   a. correct labels?
#   b. correct number of labels?
# 9. Double check what should be handed in:
#   a. the code used (can be .Zip, but no other compression)
#   b. the .txt file with ONLY the labels inferred from the evaluation data (ALL LABELS MUST BE AS IN TRAINING DATA)

# Model selection
# https://towardsdatascience.com/the-beginners-guide-to-selecting-machine-learning-predictive-models-in-python-f2eb594e4ddc
# https://scikit-learn.org/stable/computing/scaling_strategies.html#scaling-strategies
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection 


######### New thoughts after some iterations: ###############
#   PCA or automatic feature selection could be interesting?
#   Should try some more models
#   Test dividing data by "methods" - {GMMs and Accordions, Bayesian Inference}
#   Getting feature correlations through pandas # USING THIS NOW
#   Cross-validation of models # USING THIS NOW
#   Using some sort of grid search for tuning hyperparameters
#   Extracting the "best" parameters for use in the final model

# Findings: x2 & x11 have 100% correlation! -> drop one? (DROPPED x11)

def processData(): # Help function
    '''Processes data from .csv file and returns features and labels separately '''
    # using pandas to read CSV: https://www.youtube.com/watch?v=fmRm85yy5zE&ab_channel=MonkHaus

    filename = 'TrainOnMe.csv'
    headers = ['','y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'] #the headers used for columns
    df = pd.read_csv(filename, names = headers, header = 0, usecols= headers[1:14]) # create dataframe from .csv file #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    print(df.head()) # print the head of the module to see that data is correct
    
    
    classes = np.unique(df['y']) # retrive unique classes
    methods = np.unique(df['x6']) # retrive unique "methods"
    df.y = pd.Categorical(df.y) # change column y to categorical
    df['y'] = df.y.cat.codes # convert labels to numbers to get correlation
    df.x6 = pd.Categorical(df.x6)
    df['x6'] = df.x6.cat.codes

    #print("classes", classes) # print classes ['Atsuto' 'Bob' 'Jorg' 'Shoogee'] = [0, 1, 2, 3]
    #print("x6 methods: ", methods) # print methods ['Bayesian Inference', 'GMMs and Accordions']
    
    #plotData(df) # plots correlations in heatmap
    
    df = df.drop(['x11'], axis = 1) # , 'x2', 'x3', 'x5', 'x7', 'x8', 'x10' drop features after heatmap correlation analysis x11 since it correlates perfectly with x2!
    print(df.head()) 

    X = np.array(df.iloc[:,1:15]) # get all feature columns 1-15 (added split of Bayesian Inference and GMMs and Accordions)
    Y = np.array(df['y']) # get all labels in column named 'y'

    return X, Y, classes

def processEvalData():
    '''Processes data from .csv file and returns features and labels separately '''
    # using pandas to read CSV: https://www.youtube.com/watch?v=fmRm85yy5zE&ab_channel=MonkHaus

    filename = 'EvaluateOnMe.csv'
    headers = ['', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'] #the headers used for columns
    df = pd.read_csv(filename, names = headers, header = 0, usecols= headers[1:14]) # create dataframe from .csv file #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    print(df.head()) # print the head of the module to see that data is correct
    
    df.x6 = pd.Categorical(df.x6)
    df['x6'] = df.x6.cat.codes
    
    #plotData(df) # plots correlations in heatmap
    df = df.drop(['x11'], axis = 1) # , 'x2', 'x3', 'x5', 'x7', 'x8', 'x10' drop features after heatmap correlation analysis: x11 since it correlates perfectly with x2!
    print(df.head()) 
    X = np.array(df.iloc[:,0:15]) # get all feature columns 0-15 (added split of Bayesian Inference and GMMs and Accordions)

    return X

def plotData(df): # help function heatmap correlation
    plt.figure(figsize=(13,13))
    cor = df.corr()
    print(cor)
    sns.heatmap(cor, annot=True, cmap="RdYlGn") #cmap = plt.cm.Reds 
    plt.show()

def writeToTxt(predictions, classes): # Help function
    '''takes in data that will be written to a file''' # https://www.codegrepper.com/code-examples/python/python+write+a+list+to+a+file+line+by+line
    
    filename = input('Name file to write to: ') + '.txt' # name the file where labels should be written

    with open(filename, 'w') as file:
        i = 1
        for row in predictions:
            #print("row number: ", i, "row value: ", row, "class equivalent :", classes[row]) 
            file.write(classes[row] + '\n')    # Fix so that there NO EMPTY ROW AT THE END!
            i += 1
    print('Done writing to file')

def randomForestClassifyer(X,Y, k):
    '''takes in datapoints X and labels Y to create model''' #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.score
    
    rFC = cross_val_score(RandomForestClassifier(max_depth=2, random_state=None), X, Y, cv = k)
    print("Random Forest cross validation accuracy: \n ", rFC, "\n average: ", rFC.mean(), "\n")

def naiveBayesClassifyer(X,Y, k): #https://scikit-learn.org/stable/modules/naive_bayes.html
    '''Naive bayes classifyer, has issues with text labels'''
    
    NBC = cross_val_score(GaussianNB(), X, Y, cv = k)
    print("Gaussian Naive Bayes cross validation accuracy: \n", NBC, "\n average: ", NBC.mean(), "\n")

    #cnb = CategoricalNB() # Negative values does not work for this one 
    #clf = ComplementNB() # Negative values does not work for this one
    # nnb = MultinomialNB() # Negative values does not work for this one

def kNearestNeighbourClassifyer(X,Y, k): # grid search https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8&ab_channel=DataSchool
    k_range = range(1,31)
    weight_options = ['uniform', 'distance']
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv = k, scoring = 'accuracy')
    grid.fit(X, Y)

    print("kNearestNeigbour grid search best score: \n", grid.best_score_, " \n best params: ", grid.best_params_, "\n")
    
    param_dist = dict(n_neighbors = k_range, weights = weight_options)
    rand = RandomizedSearchCV(knn, param_dist, cv = k, scoring = 'accuracy', n_iter = 10)
    rand.fit(X, Y)
    print("kNearestNeigbour randomized search best score: \n", rand.best_score_, " \n best params: ", rand.best_params_, "\n")

    knn = cross_val_score(KNeighborsClassifier(n_neighbors= 5), X, Y, cv = k)
    print("k-Nearest neighbour cross validation: \n", knn, "\n average: ", knn.mean() , "\n")

def gradientBoostingClassifyer(X,Y, k): # doc: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.get_params
    gbc = cross_val_score(GradientBoostingClassifier(n_estimators=180, learning_rate= 0.1, max_features ='auto'), X, Y, cv = k)
    print("Gradient Boosting cross validation accuracy: \n", gbc, "\n average: ", gbc.mean() , "\n")

def supportVectorClassifyer(X,Y, k):
    svm = cross_val_score(SVC(kernel = 'rbf'), X, Y)
    print("Support Vector Classifier cross validation accuracy: \n", svm, "\n average: ", svm.mean(), "\n")

def logisticRegressionCV(X,Y,k):
    lrCV = cross_val_score(LogisticRegression(), X, Y)
    print("Logistic Regression cross validation accuracy: \n", lrCV, "\n average: ", lrCV.mean(), "\n" )

def adaBoostClassifier(X, Y, k):
    adaB = cross_val_score(AdaBoostClassifier(n_estimators= 180), X, Y, cv = k)
    print("AdaBoost cross validation accuracy: \n", adaB, "\n average:", adaB.mean(), "\n")

def passiveAggressiveClassifier(X, Y, k):
    pAC = cross_val_score(PassiveAggressiveClassifier(), X, Y, cv = k)
    print("Passive Agressive Classifier cross validation accuracy: \n", pAC, "\n average: ", pAC.mean() , "\n")

def mlpClassifier(X, Y, k): # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
    MLPC = cross_val_score(MLPClassifier(max_iter = 100000), X, Y, cv = k)
    print("MLP Classifier cross validation accuracy: \n", MLPC, "\n average: ", MLPC.mean(), "\n")

def decisionTreeClassifier(X, Y, k):
    dtc = cross_val_score(DecisionTreeClassifier(), X, Y, cv = k)
    print("Decision tree cross validation accuracy: \n", dtc, "\n average: ", dtc.mean() , "\n")


def XGBoost(X, Y, k): # https://towardsdatascience.com/getting-started-with-xgboost-in-scikit-learn-f69f5f470a97
    #https://xgboost.readthedocs.io/en/latest/parameter.html

    xgb = cross_val_score(XGBClassifier( use_label_encoder = False, eval_metric = 'merror', num_class = 4), X, Y, cv= k, scoring='accuracy')
    print("XGBoost cross validation accuracy: \n", xgb, " \n average: ", xgb.mean() , "\n")
    
    # Randomized search
    booster_options = ['gbtree', 'gblinear', 'dart']
    eta_options = [x /20.0 for x in range(0,10) ] #https://stackoverflow.com/questions/7267226/range-for-floats
    gamma_options = range(0,1,10)
    depth_options = range(4,10)
    child_weight_options = range(0,10)
    subsample_options = [x /10.0 for x in range(0,10) ]

    
    param_dist = dict(eta = eta_options, gamma = gamma_options, max_depth = depth_options, #booster = booster_options,
    min_child_weight = child_weight_options, subsample = subsample_options)
    rand = RandomizedSearchCV(XGBClassifier(use_label_encoder = False, eval_metric = 'merror', num_class = 4), param_dist, cv = k, scoring = 'accuracy', n_iter = 10)
    rand.fit(X, Y)
    print("XGBoost randomized search best score: \n", rand.best_score_, " \n best params: ", rand.best_params_, "\n")
    

    # Final prediction
    model = rand.best_estimator_ # get the best model to fit data on
    #confusion(model, X, Y) # confusion matrix for best model

    model.fit(X,Y)
    X = processEvalData()
    predictions = model.predict(X)

    return predictions
    #confusion_matrix

def feature_selection(X, Y): #visualisation of features: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
    model = ExtraTreesClassifier()
    model.fit(X,Y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x12'])
    feat_importances.nlargest(11).plot(kind='barh')
    plt.show()

def confusion(model, X,Y): # help function confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.8, random_state=0)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    print(confusion_matrix(y_true= y_test, y_pred= pred))

def checkGT(): # Used to check the accuracy when Ground Truth became avalible!
    file1 = "classified.txt"
    file2 = "EvaluationGT.csv"

    header = ['y']
    df1 = pd.read_csv(file1, names=header) 
    df2 = pd.read_csv(file2, names = header)
    
    Y1 = np.array(df1['y'])
    Y2 = np.array(df2['y'])

    print(len(np.where(Y1 == Y2)[0]) / len(Y2))
    


def main(): # creating different classifiers: https://www.youtube.com/watch?v=L9BLLWRtnOU&ab_channel=KindsonTheTechPro
    X, Y, classes = processData() # process data from .csv to separate features and labels

    K = 10 # Number of folds for cross-validation
    #principalComponentAnalysis(X) # (FIX PRINCIPAL COMPONENT ANALYSIS)
    #feature_selection(X, Y) # (Not that useful (unlikely linear relationship between features and data))

    #randomForestClassifyer(X,Y,K)
    #naiveBayesClassifyer(X,Y,K)
    #kNearestNeighbourClassifyer(X,Y,K)
    #gradientBoostingClassifyer(X,Y,K) # Currently one of the best at 69.9% 
    #supportVectorClassifyer(X, Y, K)
    #logisticRegressionCV(X, Y, K)
    #adaBoostClassifier(X,Y, K)
    #passiveAggressiveClassifier(X,Y,K)
    #mlpClassifier(X,Y, K)
    #decisionTreeClassifier(X,Y,K)
    classified = XGBoost(X,Y, K) # Currently one of the best at 72.7% # XGBoost randomized search best score: 0.729080808080808, best params:  {'subsample': 0.6, 'min_child_weight': 0, 'max_depth': 6, 'gamma': 0, 'eta': 0.2} 
    
    # REMEMBER TO TRAIN THE FINAL MODEL ON ALL THE DATA USING OPTIMAL PARAMETERS!
    writeToTxt(classified, classes) # use returned predictions from best model and write to .csv

if __name__ == "__main__":
    #main()
    checkGT()


########### BIN WITH TEST CODE #############

#grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores]
    #plt.plot(k_range, grid_mean_scores)
    #plt.ylabel("grid mean scores")
    #plt.xlabel("k_range")
    #plt.show()



#def principalComponentAnalysis(X): # Do principal component analysis to reduce number of features?
#    pca = PCA(svd_solver='auto')
#    pca.fit(X)
#    nX = pca.transform(X)
#    print(nX)

#def GaussianMixtureModel(X, Y, k): # DOES NOT WORK PROPERLY, NEED TO UNDERSTAND MORE: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
#    GMM = cross_val_score(GaussianMixture(), X, Y, cv = k)
#    print("Gaussian Mixture Model cross validation accuracy: \n", GMM, "\n average: ", GMM.mean(), "\n")


#method = pd.get_dummies(df['x6'])
#labels = pd.get_dummies(df['y'])
#df = pd.concat((df, method, labels), axis = 1 ) # concatenates dummies with matrix
#df = df.drop(['x6'], axis = 1) # drop column which is not needed any more
#df = df.drop(['y'], axis = 1)
    

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= split, random_state=0)
#neigh = KNeighborsClassifier(n_neighbors=5)
#neigh.fit(X_train, y_train)
#print("Nearest neighbour accuracy: ", neigh.score(X_test,y_test))

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= split, random_state=0)
#clf = RandomForestClassifier(max_depth=2, random_state=0)
#clf.fit(X_train, y_train)
#print("Random Forest Classifyer accuracy: ", clf.score(X_test, y_test))
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= split, random_state=0)

#gnb = GaussianNB()
#gnb.fit(X_train, y_train)
#print("Gaussion Naive Bayes Classifyer accuracy: ", gnb.score(X_test, y_test))

# using genfromtxt https://stackoverflow.com/questions/3518778/how-do-i-read-csv-data-into-a-record-array-in-numpy
    #   https://stackoverflow.com/questions/27066371/genfromtxt-returning-nan-rows
    

  #print(X[0])
    #print(Y)


    #data = np.genfromtxt(filename, delimiter = ',', skip_header = 0, dtype=None, encoding = 'utf-8', invalid_raise= True, names = True) # seems like it works
    #print(data.dtype.names) # checking that the first row of the matrix is correct
    
    #Npts = data.shape[0]
    #Ndims = len(data.dtype.names)
    #print(Npts, Ndims)

    #for j, colname in enumerate(data.dtype.names):
    #    print(j, colname)
        # matrix[j] = data[colname]
    
    #print(data['i']) # it is possible to access the columns by name
    #print(data[0]) # checking first row to see that it worked


###############################################################