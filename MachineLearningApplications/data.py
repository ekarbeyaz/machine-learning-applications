from PyQt5.QtWidgets import QMessageBox

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer, label_binarize
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, auc, roc_curve, classification_report, r2_score, accuracy_score, explained_variance_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns

#Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, svm
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
from itertools import cycle

class dataclass:
    #DATAFRAME OPERATIONS
    """
    Imports the dataset by given file path. Returns the data as a DataFrame.
    """
    def read_csv(self,filepath):
        return pd.read_csv(str(filepath)) 
    
    """
    Returns the columns of the given DataFrame as a List.
    """
    def get_column_list(self, df):
        column_list=[]
        for i in df.columns:
            column_list.append(i)
        return column_list
    
    """
    Returns the columns with null values of the given DataFrame as a List.
    """
    def get_empty_list(self, df):
        empty_list=[]
        for i in df.columns:
            if(df[i].isnull().values.any()==True):
                empty_list.append(i)
        return empty_list
    
    """
    Returns the columns with categorical data of the given DataFrame as a List.
    """
    def get_cat_list(self, df):
        cat_col_list=[]
        for i in df.columns:
            if(df[i].dtype=='object'):
                cat_col_list.append(i)
        return cat_col_list
    
    """
    Returns the shape of the given DataFrame.
    """
    def get_df_shape(self, df):
        return df.shape
        
    #FEATURE SELECTION
    """
    This function applies feature selection algorithms to a given DataFrame. Separates the given DataFrame into X and y according to the given target column. 
    If the index value is 0, the function applies the SelectKBest algorithm. If it's 1, it applies the SelectPercentile algorithm. After this operation, 
    it merges the new X and y values and returns it as a DataFrame.
    """
    def select_features(self, df, target_value, index, value):
        X=df.drop(columns=target_value)   
        y=df[target_value]
        y=y.to_frame()
        
        if index==0:
            selector = SelectKBest(chi2, k=int(value))      
        elif index==1:
            selector = SelectPercentile(chi2, percentile=int(value))
            
        selector.fit(X, y)
        X_new = selector.transform(X)
        new_columns=X.columns[selector.get_support(indices=True)].tolist()             
        X_new= pd.DataFrame.from_records(X_new)
        X_new.columns=new_columns
        df=pd.concat([X_new, y], axis=1)
        return df
    
    #DATA TRANSFORMATION
    """
    It performs a categorical transformation on a given column. Then returns the DataFrame.
    """
    def conv_categorical(self, df, column):
        le=LabelEncoder()
        df[column] = le.fit_transform(df[column])
        return df
        
    """
    It performs a data transformation on a given DataFrame.
    """
    def convert_features(self, df, target, func):
        if func=="MinMaxScale":
              scaler=MinMaxScaler()
        elif func=="StandardScale":
              scaler=StandardScaler()
        elif func=="Normalizer":
              scaler=Normalizer()
        elif func=="PCA":
              scaler=PCA()
        x=df.drop(target,axis=1)
        scaled_features=scaler.fit_transform(x)
        scaled_features_df = pd.DataFrame(scaled_features, index=x.index, columns=x.columns)
        scaled_features_df[target]=df[target]
        return scaled_features_df
    
    #IMBALANCED OPERATIONS
    """
    Returns the number of categories in the entered target column. Takes DataFrame and target column as parameters. Obtains the X and y values using the target column. 
    Calculates the number of categories in the resulting target space and returns it.
    """
    def get_sampling(self, df, target_value):
        X=df.drop(columns=target_value)   
        y=df[target_value]
        y=y.to_frame()
        
        X=X.values
        y=y.values
        y=y.reshape(-1,)    
        
        return Counter(y)
     
    """
    Applies UnderSampling or OverSampling to the DataFrane according to the entered index value. It takes DataFrame, target attribute, and index value as parameters.
    It obtains the X and y values from the DataFrame with the entered column name. Selects and applies the algorithm according to the index value. Finally, it returns the DataFrame.
    """
    def apply_sampling(self, df, target_value, index):
        columns = df.columns
        
        X=df.drop(columns=target_value)   
        y=df[target_value]
        y=y.to_frame()
        
        X=X.values
        y=y.values
        y=y.reshape(-1,)  
        
        print("Before " ,Counter(y))
        
        if index == 0:
            sampling = RandomUnderSampler() 
        elif index == 1:
            sampling = RandomOverSampler()
        
        X, y = sampling.fit_resample(X, y)
        print("After " ,Counter(y))
        y = y.reshape(-1, 1)        
        X = np.concatenate((X,y),axis=1)     
        
        df = pd.DataFrame.from_records(X)
        df.columns = columns
        
        return df        
    
    #NULL OPERATIONS
    """
    Deletes the column from the entered DataFrame, and returns the DataFrame.
    """
    def drop_column(self, df, column):
        return df.drop(column,axis=1)
    
    """
    Fills the null values in the entered column with the average values in that column. It takes DataFrame and column name as parameters.
    The type of the entered column is kept in a variable and this value is checked with the if else condition. If another type of data is 
    entered than its own type, it will give a type error. If there is no error, the column will be fill with new data and the DataFrame is returned.
    """
    def apply_null_mean(self, df, column):
        columnType=str(df[column].dtype)
        if columnType=="float64" or columnType=="int64":
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error.")
            msg.setWindowTitle("Error")
            msg.exec_()
        return df
    
    """
    Fills the null values in the entered column with the median values in that column. It takes DataFrame and column name as parameters.
    The type of the entered column is kept in a variable and this value is checked with the if else condition. If another type of data is 
    entered than its own type, it will give a type error. If there is no error, the column will be fill with new data and the DataFrame is returned.
    """
    def apply_null_median(self, df, column):
        columnType=str(df[column].dtype)
        if columnType=="float64" or columnType=="int64":
            df[column].fillna(df[column].median(), inplace=True)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error.")
            msg.setWindowTitle("Error")
            msg.exec_()
        return df
    
    """
    It updates the null values in the entered column with the entered value. It takes DataFrame and column name as parameters.
    The type of the entered column is kept in a variable and this value is checked with the if else condition. If another type of data is 
    entered than its own type, it will give a type error. If there is no error, the column will be fill with new data and the DataFrame is returned.
    """    
    def apply_null_value(self, df, value, column):
        columnType=str(df[column].dtype)
        if columnType=="float64":
            df[column].fillna(float(value), inplace=True)
        elif columnType=="int64":
            df[column].fillna(int(value), inplace=True)
        elif columnType=="object":
            df[column].fillna(value, inplace=True)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Type Error.")
            msg.setWindowTitle("Error")
            msg.exec_()
        return df   
    
    """
    Creates and returns X and y values from the entered DataFrame.
    """
    def create_X_y(self, df, target_value):
        X=df.drop(columns=target_value)   
        y=df[target_value]
        y=y.to_frame()        
        return X, y
    
    """
    It applies the Hold-Out operation to the entered DataFrame. It takes DataFrame, column name, percent and shuffle value as parameters.
    Creates X and y values with using create_X_y function. Training and test separation is performed using these x and y values.
    The results are added to a list and returned.
    """
    def split_hold_out(self, df, target_value, percent, shuffle): 
        split_list=[]
        if percent>=1 and percent<=99:
            X, y = self.create_X_y(df, target_value)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(percent*0.01), shuffle=shuffle)
            split_list.append([X_train, y_train, X_test, y_test])
            return split_list  

    """
    It applies the K-Fold operation to the entered DataFrame. It takes DataFrame, column name, shuffle and k value as parameters.
    Creates X and y values with using create_X_y function. Training and test separation is performed using these x and y values.
    The results are added to a list and returned.
    """
    def split_k_fold(self, df, target_value, shuffle, k_fold_count):
        split_list=[]    
        X, y = self.create_X_y(df, target_value)
        kf = KFold(n_splits=k_fold_count, shuffle=shuffle)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            split_list.append([X_train, y_train, X_test, y_test])
        return split_list
    
    """
    Training and testing of the selected model. It takes Index value of the selected model from ComboBox, KNN value and a list that contains X and y values as parameters.
    The model is selected according to the entered index value. Then X and y values are assigned to variables. Training and testing of the model is performed.
    Results are kept in a list and returned. These lists will then be used for score and graph operations.
    """
    def train_test_model(self, index, value, split_list):
        model_type = 0
        if index==0:
            model = LinearRegression()
            model_type=1
        elif index==1:
            model=tree.DecisionTreeClassifier()
            model_type=0
        elif index==2:
            model=tree.DecisionTreeRegressor()
            model_type=1
        elif index==3:
            model=KNeighborsClassifier(n_neighbors=value)
            model_type=0
        elif index==4:
            model=KNeighborsRegressor(n_neighbors=value)
            model_type=1
        elif index==5:
            model=GaussianNB()
            model_type=0
        elif index==6:
            model=svm.SVC(probability=True)
            model_type=0
            
        X_test_list = []
        y_test_list=[]
        predict_list=[]
        
        for i in range(len(split_list)):
            X_train, y_train, X_test, y_test=split_list[i][0], split_list[i][1], split_list[i][2], split_list[i][3]
            X_train=X_train.to_numpy()
            y_train=y_train.to_numpy()
            X_test=X_test.to_numpy()
            y_test=y_test.to_numpy()
            y_train=y_train.flatten()
            y_test=y_test.flatten()
            
            model.fit(X_train, y_train)
            prediction=model.predict(X_test)
                          
            X_test_list.append(X_test)
            y_test_list.append(y_test)
            predict_list.append(prediction)
        return X_test_list, y_test_list, predict_list, model_type
    
    """
    Calculation of metrics and confusion matrices. If K-Fold is applied, the calculation process will be applied for each K value. It takes the X and y values, 
    predicted values and model type returned from the training and testing process as parameters. Using these lists, values such as the success of the model and 
    the confusion matrix are calculated. After this process the obtained each score is stored in its own list and returned. If the model type is 0, the confusion matrix
    and model success are calculated for the classification models. If its 1, metrics of the regression models such as r2 scale are calculated.
    """
    def metrics(self, X_test_list, y_test_list, predict_list, model_type):
        accuracy_score_list = []
        score_list = []
        conf_list = []
        for i in range(len(X_test_list)):
            if model_type==0:
                score=accuracy_score(predict_list[i], y_test_list[i])
                report=classification_report(y_test_list[i], predict_list[i])
                
                score_str = "Accuracy Score : %"+str(score*100)[:5]+"\n"+str(report)
                score_list.append(score_str)
                
                conf_matrix = confusion_matrix(y_test_list[i], predict_list[i])
                conf_list.append(conf_matrix)
                
                acc_str = "Accuracy Score: %"+str(score*100)[:5]
                accuracy_score_list.append(acc_str)
                
            elif model_type==1:
                r2score=r2_score(y_test_list[i], predict_list[i])
                variancescore=explained_variance_score(y_test_list[i], predict_list[i])
                mae = mean_absolute_error(y_test_list[i], predict_list[i])
                mse = mean_squared_error(y_test_list[i], predict_list[i], squared = False)
                
                score_str = "R2 Score: %"+str(r2score*100)[:5]+"\n"+"Variance Score: %"+str(variancescore*100)[:5]+"\n"+"Mean Absolute Error: %"+str(mae*100)[:5]+"\n"
                score_str = score_str+"Mean Squared Error: %"+str(mse*100)[:5]+"\n"
                score_list.append(score_str)
                
                acc_str = "R2 Score: %"+str(r2score*100)[:5]
                accuracy_score_list.append(acc_str)
        return score_list, conf_list, accuracy_score_list
     
    """
    It takes the list of confusion matrices calculated in the metric function as a parameter. If Hold-Out is applied, it creates the confusion matrix only once.
    If K-Fold is applied, it creates a confusion matrices and an overlapped matrix equal to the K value. The created matrices are saved to a file and the file paths
    are then returned to be displayed on the screen for later.
    """
    def plot_cm_list(self, conf_list):
        path_list = []
        title_list =[]
        
        for i,cm in enumerate(conf_list):
            path="./plotimages/cf_"+str(i+1)+".png"
            title = "Confusion Matrix "+str(i+1)
            fig = plt.figure()
            sns.heatmap(cm, annot=True, cmap='Blues', fmt = ".1f")
            plt.show()
            fig.savefig(path)
            path_list.append(path)
            title_list.append(title)
        
        ov_path=""
        ov_title=""
        if len(conf_list) != 1:
            ov_path = "./plotimages/cf_overlapped.png"
            ov_title = "Overlapped Matrix"
            overlapped_cm=np.sum(conf_list,axis=0).astype(np.int16)
            fig = plt.figure()
            sns.heatmap(overlapped_cm, annot=True, cmap='Blues', fmt = ".1f")
            plt.show()
            fig.savefig(ov_path)
            
        return ov_path, ov_title, path_list, title_list
    
    """
    It takes the prediction list and the list of actual values formed after the prediction process of the model as parameters. Creates a graph of actual values and predicted 
    values using these lists. If K-Fold is applied, a graphic equal to the K value is created and saved in a file. This file path is then returned to be shown on the screen for later.
    """
    def plot_true_pred(self, predict_list, y_test_list):
        path_list = []
        title_list =[]
        for i,j in enumerate(y_test_list):
            path="./plotimages/pred_true_"+str(i+1)+".png"
            title = "True vs Predicted Values "+str(i+1)
            
            t, p = y_test_list[i], predict_list[i]
            fig = plt.figure()
            plt.plot(t, color = 'red', label = 'True Values')
            plt.plot(p, color = 'blue', label = 'Predicted Values')
            plt.title('Prediction')
            plt.xlabel('Length')
            plt.ylabel('Values')
            plt.legend()
            plt.show()
            fig.savefig(path)
            
            path_list.append(path)
            title_list.append(title)
        return path_list, title_list
    
    """
    It takes the prediction list and the list of actual values formed after the prediction process of the model as parameters. Builds the ROC curve using these lists.
    If K-Fold is applied, a graphic equal to the K value is created and saved in a file. This file path is then returned to be shown on the screen for later.
    """
    def plot_roc(self, predict_list, y_test_list):
        path_list = []
        title_list =[]
        
        for i,j in enumerate(y_test_list):
            path="./plotimages/roc_curve_"+str(i+1)+".png"
            title = "Roc Curve "+str(i+1)
            y_test = y_test_list[i]
            y_pred = predict_list[i]
            
            
            classes = np.unique(y_pred)            
            y_test = label_binarize(y_test, classes=classes.tolist())
            y_pred = label_binarize(y_pred, classes=classes.tolist())
                        
            lw = 2
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
                
            for i in range(len(classes)):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            fig = plt.figure(1)
            colors = cycle(['aqua', 'darkorange', 'red'])
            for i, color in zip(range(len(classes)), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                          label='ROC curve of class {0} (area = {1:0.2f})'
                          ''.format(i, roc_auc[i]))
            
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()
            fig.savefig(path)
            
            path_list.append(path)
            title_list.append(title)

        return path_list, title_list   
    
    """
    The function that makes ensemble learning. It takes the selected algorithm as parameter and data separated by Hold-Out or K-Fold.
    Performs ensemble learning to the selected algorithm. Returns the success of the model, predicted values, actual values, and the model itself.
    """
    def ensemble_(self, index, split_list):
        predict_list = []
        y_test_list = []
        score_list = []
        for i in range(len(split_list)):
            X_train, y_train, X_test, y_test = split_list[i][0], split_list[i][1], split_list[i][2], split_list[i][3]
            X_train=X_train.to_numpy()
            y_train=y_train.to_numpy()
            X_test=X_test.to_numpy()
            y_test=y_test.to_numpy()
            y_train=y_train.flatten()
            y_test=y_test.flatten() 
            
            if index == 0:
                "VOTING TAKES MORE THAN ONE CLASSIFIER"
                estimators = [('lr', LogisticRegression(multi_class='multinomial', random_state=1)), ('dtc', tree.DecisionTreeClassifier()), ('gnb', GaussianNB())]
                ensemble = VotingClassifier(estimators=estimators, voting='hard')
            elif index == 1:
                "STACKING TAKES MORE THAN ONE CLASSIFIER"
                estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)), ('svr', svm.LinearSVC(random_state=42))]
                ensemble = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
            elif index == 2:
                "BAGGING TAKES ONLY ONE CLASSIFIER"
                ensemble = BaggingClassifier(base_estimator=svm.SVC(probability=True), n_estimators=10, random_state=0)
            elif index == 3:
                "BOOSTING DOESN'T TAKE ANY CLASSIFIER"
                ensemble = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
                
            ensemble.fit(X_train, y_train)            
            prediction=ensemble.predict(X_test)
            
            score=ensemble.score(X_test, y_test)
            
            predict_list.append(prediction)
            y_test_list.append(y_test)
            score_list.append(score)
            
            return score_list, predict_list, y_test_list, ensemble
    
    """
    Grid Search vs Random Search operations. It takes the index of the model selected as the parameter, the file path of the data set, the target attribute and the state variable.
    """
    def grid_random_search(self, index, file_path, target_value, case):
        df = pd.read_csv(file_path)
            
        X=df.drop(columns=target_value)   
        y=df[target_value]
        y=y.to_frame()
        
        X=X.values
        y=y.values
        y=y.reshape(-1,)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        predict_list = []
        y_test_list = []
        
        #GridSearch
        if case == 0:                               
            if index==0:
                model=tree.DecisionTreeClassifier()
                grid_params = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
            elif index==1:
                model=KNeighborsClassifier(n_neighbors=5)
                grid_params = {'n_neighbors': [3,5,11,19],
                            'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan']}
            elif index==2:
                model=svm.SVC(probability=True)
                grid_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                          'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            
            grid_search = GridSearchCV(model, grid_params, verbose = 1, cv =3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            prediction=grid_search.predict(X_test)
            
            best_param = grid_search.best_params_ 
            best_score = grid_search.best_score_
            
                        
            predict_list.append(prediction)
            y_test_list.append(y_test)
            
            return best_param, best_score, predict_list, y_test_list
        
        elif case == 1:
            #RandomSearch
            if index==0:
                model=tree.DecisionTreeClassifier()
                grid_params = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
            elif index==1:
                model=KNeighborsClassifier(n_neighbors=5)
                grid_params = {'n_neighbors': [3,5,11,19],
                            'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan']}
            elif index==2:
                model=svm.SVC(probability=True)
                grid_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                          'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
            
            rand_search = RandomizedSearchCV(model, grid_params, cv=10, scoring='accuracy', n_iter=10, random_state=5)
            rand_search.fit(X_train, y_train)
            prediction=rand_search.predict(X_test)
            
            best_param = rand_search.best_params_ 
            best_score = rand_search.best_score_
            
            predict_list.append(prediction)
            y_test_list.append(y_test)
            
            return best_param, best_score, predict_list, y_test_list
        
    