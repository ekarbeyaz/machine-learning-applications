import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QLabel, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi

from sklearn.metrics import confusion_matrix

import pandas as pd

import data, displaytable
        
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("./Designs/mainwindow.ui", self)
        
        global data
        data=data.dataclass()
        
        self.popups = []
         
        #Main Screen Operations   
        self.listW.clicked.connect(self.get_target)
        
        self.btn_import.clicked.connect(self.import_data)        
        self.btn_target.clicked.connect(self.set_target)
        self.btn_con_cat.clicked.connect(self.convert_categorical)
        self.btn_feature_select.clicked.connect(self.feature_selection)
        self.btn_feature_transform.clicked.connect(self.feature_transformation)
        self.btn_imbalanced.clicked.connect(self.imbalanced_data)
        self.btn_drop.clicked.connect(self.drop_column)
        self.btn_mean.clicked.connect(self.null_mean)
        self.btn_median.clicked.connect(self.null_median)
        self.btn_null_value.clicked.connect(self.null_value)
        self.btn_cancel.clicked.connect(self.cancel)
        
        #Hold-Out ve K-Fold Tab Operations
        self.btn_holdout.clicked.connect(self.hold_out)
        self.btn_kfold.clicked.connect(self.k_fold)
        self.cb_kfold.currentIndexChanged.connect(self.show_k_fold)
        
        #Classification and Test Tab Operations
        self.btn_train.clicked.connect(self.train_test_model)
        self.btn_score.clicked.connect(self.clear_score)
        self.btn_confusion.clicked.connect(self.conf_matrix)
        self.btn_true_predict.clicked.connect(self.pred_true)
        self.btn_roc.clicked.connect(self.roc_curve)
        
        #Search Tab Operations
        self.btn_grid.clicked.connect(self.grid_search)
        self.btn_random.clicked.connect(self.random_search)
        self.btn_search_cf.clicked.connect(self.show_cf_search)
        self.btn_search_cf_2.clicked.connect(self.show_cf_search)
        
        #Ensemble Tab Operations
        self.btn_ensemble.clicked.connect(self.ensemble_)
        self.btn_ensemble_cf.clicked.connect(self.show_cf_ensemble)
     
    """
    --------------------------------------------------------------------------------------
    Main Window
    --------------------------------------------------------------------------------------
    """
    """
    Opens the file import screen. The dataset is selected. The file path of the selected data is transferred to a variable. The details function to be 
    imported is called by using this address.
    """
    def import_data(self):
        self.filePath, _ = QFileDialog.getOpenFileName(self, 'Select Dataset',"./databases","csv(*.csv)")
        if self.filePath!="":
            self.details(0)
            
    """
    Updates the Widgets on the screen. Also, the imports the dataset with given file path. For import, the file path is sent to the read_csv function in 
    the data class, and the data read back from the function is returned as DataFrame. Column information of the data set is taken from the data class and 
    written to the ListWidget object. This object shows the rows and column types in the data set. Also, it creates a model from the DataFrame by sending 
    the DataFrame returned from the data class to the class named displaytable. With this model, it displays the data in the ListView object on the screen.
    """
    def details(self, case=1):
        if case==0:
            self.item = ""
            self.target_value=""
            self.lbl_target.setText(self.target_value)
            self.le_feature_select.setText("3")
            self.df=data.read_csv(self.filePath)  
            self.split_details()
        
        self.clear_items() 
        self.column_list=data.get_column_list(self.df)
        self.cat_col_list=data.get_cat_list(self.df)
        self.feature_select_list=["SelectKBest","SelectPercentile"]
        self.scale_list=["MinMaxScale","StandardScale","Normalizer","PCA"]
        self.sampling_list=["UnderSampling","OverSampling"]
        self.empty_list=data.get_empty_list(self.df)        
        
        for i ,j in enumerate(self.column_list):
            text=j+ " -------   " + str(self.df[j].dtype)
            self.listW.insertItem(i,text)
        
        df_shape="Dataset Shape: Row Count: "+ str(data.get_df_shape(self.df)[0])+"  Column Count: "+str(data.get_df_shape(self.df)[1])
        self.lbl_data_shape.setText(df_shape)
        
        model=displaytable.DataFrameModel(self.df)
        self.tbl_data.setModel(model)
        
        self.write_combobox()
        
    """
    Clears any Widgets found on the screen. In this way, the data in Widgets will not be overwritten.
    """
    def clear_items(self):
        self.le_null_value.clear()
        self.listW.clear()
        
        self.cb_cat_column.clear()
        self.cb_feature_select.clear()
        self.cb_feature_transform.clear()
        self.cb_sampling.clear()      
        self.cb_drop_column.clear()
        self.cb_empty_column.clear()
     
    """
    Prints the lists defined in Details function to the ComboBoxes on the screen.
    """
    def write_combobox(self):
        self.cb_cat_column.addItems(self.cat_col_list)
        self.cb_feature_select.addItems(self.feature_select_list)
        self.cb_feature_transform.addItems(self.scale_list)
        self.cb_sampling.addItems(self.sampling_list)
        self.cb_drop_column.addItems(self.column_list)
        self.cb_empty_column.addItems(self.empty_list)   
        
    """
    Assigns the selected column value in the ListWidget object in which the columns of the data set are shown to a variable.
    This value will then be chosen as the target attribute.
    """
    def get_target(self):
        self.item=self.listW.currentItem()
    
    """
    Selects the selected column value in QListWidget as the target attribute and assigns it to a variable. Also prints the selected 
    target subjectivity to the Label object on the screen. If a value is not selected from the ListWidget object, the application gives 
    an error message on the screen and asks us to select an object from the ListWidget.
    """
    def set_target(self):
        try:
            print()
            self.target_value=str(self.item.text()).split()[0]
            self.lbl_target.setText(self.target_value)
            self.get_class_values()            
            self.train_test_details()
            self.grid_random_details()
            self.ensemble_details()
        except AttributeError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error!. Please select a column from the list.")
            msg.setWindowTitle("Error")
            msg.exec_()         
        except IndexError:pass
        
    """
    It sends the selected column in the ComboBox containing categorical data to the data class and categorical transformation is provided.
    If there is missing data in the selected column, the application displays an error message on the screen.
    """        
    def convert_categorical(self):
        try:
            column=self.cb_cat_column.currentText()
            null_count=self.df[column].isnull().sum()
            if null_count==0:
                self.df=data.conv_categorical(self.df, column)
                self.details()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please delete null values in selected column.")
                msg.setWindowTitle("Error")
                msg.exec_()
        except AttributeError: pass
        except KeyError: pass
            
    """
    It implements the feature selection process on the data set according to the algorithm selected from the ComboBox. First, it takes the selected 
    index value from the ComboBox. It takes the K value or K ratio values that the algorithm will receive from the LineEdit object and sends these 
    values to the select_features function in the data class. It also sends the DataFrame and target attribute to the function that will be processed.
    It shows the returned DataFrame in the TableView object on the screen by calling the details function.
    """
    def feature_selection(self):
        try:
            index=self.cb_feature_select.currentIndex()
            value=self.le_feature_select.text()
            
            self.df=data.select_features(self.df, self.target_value, index, value)
            self.details()
        except AttributeError: pass
        except ValueError: pass
        except KeyError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please select target column.")
            msg.setWindowTitle("Error")
            msg.exec_()
    
    """
    It applies attribute conversion operations to the data set according to the algorithm selected from the ComboBox. If the number of empty values in 
    the whole data set is greater than zero, the application will display an error message on the screen. If there is no null value in the data set, it 
    sends the value selected from ComboBox to the convert_features function in the data class. DataFrame details function returned from the function is 
    called and printed on the ListView object on the screen.
    """
    def feature_transformation(self):
        try:
            null_count=self.df.isnull().sum()
            null_count_total=0
            for i in range(len(null_count)):
                null_count_total+=null_count[i]
            
            if null_count_total==0:
                if self.cb_feature_transform.currentText()=='MinMaxScale':
                    self.df = data.convert_features(self.df, self.target_value, "MinMaxScale")
                elif self.cb_feature_transform.currentText()=='StandardScale':
                    self.df = data.convert_features(self.df, self.target_value, "StandardScale")
                elif self.cb_feature_transform.currentText()=='Normalizer':
                    self.df = data.convert_features(self.df, self.target_value, "Normalizer")
                elif self.cb_feature_transform.currentText()=='PCA':
                    self.df = data.convert_features(self.df, self.target_value, "PCA")
                self.details()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please delete null values in selected column.")
                msg.setWindowTitle("Error")
                msg.exec_()
            
        except AttributeError: pass
        except ValueError: pass
        except KeyError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please select target column.")
            msg.setWindowTitle("Error")
            msg.exec_()
        
    """
    Applies UnderSampling and OverSampling according to the process selected from ComboBox. It sends the index value of the selected process from the ComboBox, 
    the DataFrame, and the target attribute to the apply_sampling function in the data class. It calls the returning DataFrame details function and updates it on the screen.
    """       
    def imbalanced_data(self):
        try:
            null_count=self.df.isnull().sum()
            null_count_total=0
            for i in range(len(null_count)):
                null_count_total+=null_count[i]
                
            if null_count_total==0:                
                Index = self.cb_sampling.currentIndex()
                self.df = data.apply_sampling(self.df, self.target_value, Index)            
                self.details()
                self.get_class_values() 
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Please delete null values in selected column.")
                msg.setWindowTitle("Error")
                msg.exec_()
        except AttributeError: pass
        except ValueError: pass
        except KeyError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please select target column.")
            msg.setWindowTitle("Error")
            msg.exec_()
    
    """
    Calculates the number of categories in the target column and prints to the TextBrowser object. For this operation, it sends the DataFrame and 
    target attribute to the get_sampling function in the data class. Prints the returned information to the TextBrowser object.
    """    
    def get_class_values(self): 
        class_values = data.get_sampling(self.df, self.target_value)
        self.tb_sampling.setText(str(class_values))
    
    """
    Deletes the selected column from the ComboBox containing columns. First, it takes the column name from the ComboBox and assigns it to a variable.
    It then sends the DataFrame and selected column to the drop_column function in the data class. It shows the returned DataFrame on the screen by 
    calling the details function.
    """
    def drop_column(self):
        try:
            column=self.cb_drop_column.currentText()
            if (column == self.target_value):
                self.target_value=""
                self.lbl_target.setText("")
                
            self.df=data.drop_column(self.df, column)
            self.details()     
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass      
    
    """
    Fills empty data in the selected column from the ComboBox with the average value. First, it gets the column name from the ComboBox. It then sends the 
    DataFrame and column name to the apply_null_mean function in the data class. It shows the returned DataFrame on the screen by calling the details function.
    """    
    def null_mean(self):
        try:
            column=self.cb_empty_column.currentText()
        
            self.df=data.apply_null_mean(self.df, column)
            self.details()    
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass  
    
    """
    Fills empty data in the column selected from the ComboBox containing empty columns with a median value. First, it gets the column name from the ComboBox.
    It then sends the DataFrame and column name to the apply_null_median function in the data class. It shows the returned DataFrame on the screen by calling the details function.
    """
    def null_median(self):
        try:
            column=self.cb_empty_column.currentText()
        
            self.df=data.apply_null_median(self.df, column)
            self.details()      
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass  
           
    """
    Updates the blank data in the column selected from the ComboBox containing empty columns with the data entered in the LineEdit object. First, it gets the column name 
    from the ComboBox. Gets the data to be updated from the LineEdit object. It then sends the DataFrame, data, and column name to the apply_null_value function in the data class.
    It shows the returned DataFrame on the screen by calling the details function.
    """
    def null_value(self):          
        try:
            column=self.cb_empty_column.currentText()
            value=self.le_null_value.text()
            
            self.df=data.apply_null_value(self.df, value, column)
            self.details()              
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass     
    
    """
    Undoes the operations applied on the data set, and imports the data set again.
    """
    def cancel(self):
        try:
            self.details(0)
        except AttributeError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please import a dataset first.")
            msg.setWindowTitle("Error.")
            msg.exec_()
            
    """
    --------------------------------------------------------------------------------------
    Hold-Out and K-Fold Tab
    --------------------------------------------------------------------------------------
    """
    """
    Updating the objects in the Hold-Out and K-Fold tabs.
    """
    def split_details(self):
        self.le_percent.setText("20")
        self.le_kfold.setText("3")
        self.cb_shuffle.setChecked(True)
        
    
    """
    Function of Hold-Out operation on data set.For this operation, the function split_hold_out in the data class is called, and it sends the DataFrame 
    the target attribute, test rate, and shuffling of the data to the function. A list returns from the function as training and test values. 
    Training and test values in this list are sent to the write_tableview function to be displayed on the screen.
    """
    def hold_out(self):
        try:
            self.cb_kfold.clear()
            percent=int(self.le_percent.text())
        
            shuffle = False
            if self.cb_shuffle.isChecked():
                shuffle = True
            else:
                shuffle = False
                
            self.split_list = data.split_hold_out(self.df, self.target_value, percent, shuffle)
            self.write_tableview(self.split_list, 1)           
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass 
    
    """
    Performing the K-Fold operation. It takes the K value and the data set hashing value from the related objects and assigns them to the variables. 
    It then calls the split_k_fold function in the data class and sends the DataFrame, target attribute, hash value and k value as parameters. 
    Assigns the returning list to the variable. Finally, it adds the k value to the K-Fold ComboBox.
    """
    def k_fold(self):
        try:
            self.cb_kfold.clear()  
            
            shuffle = False
            if self.cb_shuffle.isChecked():
                shuffle = True
            else:
                shuffle = False
                
            k_fold_count = int(self.le_kfold.text())
            self.split_list = data.split_k_fold(self.df, self.target_value, shuffle, k_fold_count)
            for i in range(k_fold_count):
                self.cb_kfold.addItem(str(i+1))
        except AttributeError: pass
        except ValueError: pass
        except KeyError: pass 
    
    """
    When the index value of the K-Fold ComboBox changes, this function runs and sends the previously created split_list to the write_tableview function. 
    In this way, the k value selected in the ComboBox is displayed in the TableViews on the screen.
    """                       
    def show_k_fold(self):
        fold_value = self.cb_kfold.currentIndex()
        self.write_tableview(self.split_list, fold_value+1)
    
    """
    Writes the train and test spaces returned from Hold-Out and K-Fold functions to the tableview on the screen. The i value in the Split_list [i] [0] 
    variable indicates the fold number, k value if K-Fold is applied, and 0 if Hold-Out is applied. The value 0 in the second column represents the X_train space,
    1 y_train 2 X_test, 3 y_test spaces.
    """
    def write_tableview(self, split_list, fold_value):
        count=0    
        for i in range(len(split_list)):
            if count==fold_value-1:
                #Transferring the data in the model to table views
                df=pd.DataFrame.from_records(split_list[i][0]) #numpyarray to dataframe
                model=displaytable.DataFrameModel(df)
                self.tbl_xtrain.setModel(model)
                self.lbl_xtrain.setText(str(model.rowCount())+" Row "+str(model.columnCount())+" Column")
                
                df=pd.DataFrame.from_records(split_list[i][1])
                model=displaytable.DataFrameModel(df)
                self.tbl_ytrain.setModel(model)      
                self.lbl_ytrain.setText(str(model.rowCount())+" Row") 
                
                df=pd.DataFrame.from_records(split_list[i][2])
                model=displaytable.DataFrameModel(df)
                self.tbl_xtest.setModel(model)
                self.lbl_xtest.setText(str(model.rowCount())+" Row "+str(model.columnCount())+" Column")
                
                df=pd.DataFrame.from_records(split_list[i][3])
                model=displaytable.DataFrameModel(df)
                self.tbl_ytest.setModel(model)     
                self.lbl_ytest.setText(str(model.rowCount())+" Row")  
            count+=1
            
    """
    --------------------------------------------------------------------------------------
    Classification and Prediction Tab
    --------------------------------------------------------------------------------------
    """
    def train_test_details(self):
        self.cb_model.clear()
        
        model_list=["LinearRegression","DecisionTreeClassifier","DecisionTreeRegressor","KNeighborsClassifier","KNeighborsRegressor","GaussianNB","SVC"]
        for i in model_list:
            self.cb_model.addItem(i)  
        self.lbl_target_train.setText(self.target_value) 
        self.le_knn_value.setText("3")
    
    """
    Performs training and testing of the model. The index of the model selected from the Model ComboBox is kept in a variable. If the KNN model is selected, 
    the K value is taken from a LineEdit and the split_list list created after Hold-Out or K-Fold operations is sent to the train_test_model function in the data class. 
    The metric lists returned from training and testing processes are printed on the relevant Widgets on the screen.
    """
    def train_test_model(self):
        index=self.cb_model.currentIndex()
        value=int(self.le_knn_value.text())
        
        self.X_test_list, self.y_test_list, self.predict_list, self.model_type = data.train_test_model(index, value, self.split_list)
        self.score_list, self.conf_list, self.accuracy_score_list = data.metrics(self.X_test_list, self.y_test_list, self.predict_list, self.model_type)
        
        self.listW_history.clear()
        for i ,j in enumerate(self.accuracy_score_list):
            self.listW_history.insertItem(i,j)
        
        for i in range(len(self.score_list)):
            if len(self.score_list) == 1:
                self.tb_metrics.setText(str(self.score_list[i]))
            elif len(self.score_list) > 1:
                if i == 0:
                    self.tb_metrics.clear()
                prev_text = self.tb_metrics.toPlainText()
                self.tb_metrics.setText(prev_text+"K "+str(i+1)+" value : \n"+str(self.score_list[i])+"\n")

    """
    The confusion matrix list returned from the Training and Test operation is sent to the plot_cm_list function in the data class and the confusion matrices are created. 
    The file paths of the created matrices are returned and these paths are sent to a new window and displayed on the screen.
    """
    def conf_matrix(self):
        if len(self.conf_list) != 0:
            ov_path, ov_title, path_list, title_list  = data.plot_cm_list(self.conf_list)
            if len(self.conf_list) > 1:
                for i in range(len(path_list)):
                    self.open_plot(path_list[i], title_list[i])  
                self.open_plot(ov_path, ov_title)
            elif len(self.conf_list) == 1:
                for i in range(len(path_list)):
                    self.open_plot(path_list[i], title_list[i])  
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("")
            msg.setWindowTitle("Error")
            msg.exec_()
    
    """
    The predictions and facts lists returned from the Training and Test process are sent to the plot_true_pred function in the data class, and the real and 
    predicted graph is created. The file paths of the generated graphics are returned and these paths are sent to a new window and displayed on the screen.
    """
    def pred_true(self):
        path_list, title_list = data.plot_true_pred(self.predict_list, self.y_test_list)
        for i in range(len(path_list)):
            self.open_plot(path_list[i], title_list[i])
        

    """
    The predictions and facts lists returned from the Training and Test process are sent to the plot_roc function in the data class and the ROC curve graph is created. 
    The file paths of the generated graphics are returned and these paths are sent to a new window and displayed on the screen.
    """           
    def roc_curve(self):
        if len(self.conf_list) != 0:
            path_list, title_list = data.plot_roc(self.predict_list, self.y_test_list)
            for i in range(len(path_list)):
                self.open_plot(path_list[i], title_list[i])    
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("")
            msg.setWindowTitle("Error.")
            msg.exec_()

    """
    Cleaning up the QListWidget object containing the success scores.
    """
    def clear_score(self):
        self.listW_history.clear()   
        
    """
    --------------------------------------------------------------------------------------
    Ensemble Tab
    --------------------------------------------------------------------------------------
    """ 
    
    """
    Filling the ComboBox on the bulk learning page
    """
    def ensemble_details(self):
        self.cb_ensemble.clear()
        ensemble_list=["Voting", "Stacking", "Bagging", "Boosting"]
        for i in ensemble_list:
            self.cb_ensemble.addItem(i)  

    """
    Function of batch learning process according to the chosen batch learning algorithm.
    """
    def ensemble_(self):
        index = self.cb_ensemble.currentIndex()
        self.score_list, self.predict_list, self.y_test_list, ensemble = data.ensemble_(index, self.split_list)
        self.tb_ensemble.clear()
        for i in self.score_list:
            self.tb_ensemble.setText(self.tb_ensemble.toPlainText()+" Score : "+str(i)+"\n")
        self.lbl_ensemble.setText(str(ensemble))
        
    """
    The model with the best model created as a result of GridSearch and RandomSearch is trained and the confusion matrix is created. The result is shown on the matrix screen.
    """
    def show_cf_ensemble(self):
        conf_list = []
        conf_matrix = confusion_matrix(self.predict_list[0], self.y_test_list[0])
        conf_list.append(conf_matrix)
        
        ov_path, ov_title, path_list, title_list  = data.plot_cm_list(conf_list)
        for i in range(len(path_list)):
            self.open_plot(path_list[i], title_list[i]) 


    """
    --------------------------------------------------------------------------------------
    Search Tab
    --------------------------------------------------------------------------------------
    """
    def grid_random_details(self):
        self.cb_grid.clear()
        self.cb_random.clear()
        
        model_list=["DecisionTreeClassifier","KNeighborsClassifier","SVC"]
        for i in model_list:
            self.cb_grid.addItem(i)  
            self.cb_random.addItem(i)
    """
    The function that makes the GridSearch algorithm.
    """    
    def grid_search(self):
        index = self.cb_grid.currentIndex()
        best_param, best_score, self.predict_list, self.y_test_list = data.grid_random_search(index, self.filePath, self.target_value, 0)
        self.tb_grid.setText("Best Parameters: "+str(best_param)+"\n Score: "+str(best_score))
        
    """
    The function that makes the RandomSearch algorithm.
    """    
    def random_search(self):
        index = self.cb_random.currentIndex()
        best_param, best_score, self.predict_list, self.y_test_list = data.grid_random_search(index, self.filePath, self.target_value, 1)
        self.tb_random.setText("Best Parameters: "+str(best_param)+"\n Score: "+str(best_score))
        
    """
    The model with the best model created as a result of GridSearch and RandomSearch is trained and the confusion matrix is created. The result is shown on the matrix screen.
    """
    def show_cf_search(self):
        conf_list = []
        conf_matrix = confusion_matrix(self.predict_list[0], self.y_test_list[0])
        conf_list.append(conf_matrix)
        
        ov_path, ov_title, path_list, title_list  = data.plot_cm_list(conf_list)
        for i in range(len(path_list)):
            self.open_plot(path_list[i], title_list[i])  
    
    """
    --------------------------------------------------------------------------------------
    Displaying Plots on New Screen
    --------------------------------------------------------------------------------------
    """   
    """
    The file path and window name show the entered values in a new window.
    """
    def open_plot(self, path, title): #Opens the Plot page.
        self.w = Plot_window(path, title)
        self.w.show()
        self.popups.append(self.w)
        
"""
Opens a new window and prints the entered plot on the label created there.
"""            
class Plot_window(QDialog):
    def __init__(self, path, title):
        super().__init__()
        self.acceptDrops() 
        self.setWindowTitle(title)
        
        self.setGeometry(0, 0, 400, 300) 
        self.label = QLabel(self) 
        self.pixmap = QPixmap(path) 
        self.label.setPixmap(self.pixmap) 
        self.label.resize(self.pixmap.width(), self.pixmap.height()) 
        
        resolution = QDesktopWidget().screenGeometry()
        self.move((resolution.width() / 2) - (self.frameSize().width() / 2), (resolution.height() / 2) - (self.frameSize().height() / 2)) 
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())