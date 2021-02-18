# Machine Learning Applications Using PyQt5
This application performs data preprocessing and machine learning algorithms on a selected dataset using GUI.

### About this application
On the code side, this application has 3 python files. Which is "main.py", "data.py" and "displaytable.py". "main.py" is used to edit and control the widgets on the screen. In "data.py" all the algorithms and data operations are performed. "displaytable.py" is used to display the DataFrame in the QTableView widget.

In this application, we have 5 different tabs. Each tab has its own task. These tabs are Main Window, Hold-Out and K-Fold, Classification and Prediction, Ensemble and GridSearch and RandomSearch.

### Screenshots
Main Window           |  Hold-Out and K-Fold
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/78651744/108333648-3b843600-71e2-11eb-9835-f19f369d91fd.png" width="400" height="300" />  |  <img src="https://user-images.githubusercontent.com/78651744/108333666-42ab4400-71e2-11eb-9555-e1a44d4654e6.png" width="400" height="300" />

Classification and Prediction           |  Graphs
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/78651744/108333671-43dc7100-71e2-11eb-9ae0-c06e2a64915c.png" width="400" height="300" />  |  <img src="https://user-images.githubusercontent.com/78651744/108333676-44750780-71e2-11eb-9a52-50791d94b15b.png" width="400" height="100" />

Ensemble           |  GridSearch and RandomSearch
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/78651744/108333679-450d9e00-71e2-11eb-8aa2-dea9fb7a1794.png" width="400" height="300" />  |  <img src="https://user-images.githubusercontent.com/78651744/108333686-463ecb00-71e2-11eb-90a4-75c3aa2e6b20.png" width="400" height="300" />

In the Main Window tab, you can import a dataset and perform several data preprocessing techniques on that dataset. Also, you can view the imported dataset with the table on the main screen. Hold-Out and K-Fold tab are for splitting the dataset. You can split the dataset using Hold-Out or K-Fold technique. If you applied the K-Fold, you can use the Combobox for K-Fold and see each K value on the tables on the screen. In the classification tab, you can select an algorithm and train the algorithm. After the training process, the scores are will be printed below. Also, you can use buttons to view the confusion matrix, predicted and true values, and roc curve. All graphs created in this application are shown in a new window and they are saved to the "./plotimages" location. Ensemble tab you can select an algorithm and train it. Also, you can view scores and graphs after the training process. GridSearch and RandomSearch tab is the same as the ensemble tab, you can select an algorithm and train it and view scores and graphs.

Ensemble and search tabs are not very detailed. Normally in ensemble learning, you have to select an ensemble learning algorithm, and then select a couple of machine learning algorithms. You should be able to use different machine learning algorithms. But in this project, you can't do that. All ensemble algorithms have fixed machine learning algorithms. The same thing applies for the search methods. They are not detailed.

This application is not completed. It still has some errors in it. I tried my best to remove all errors while I was working on this project. But now I'm not working on it. This is the final version for me. If you encounter some errors you can try to fix them. And if you want to improve it you can do that too. I also explained every single function in this project, you can easily understand them.

### How to use it?
1. Import a dataset using the import button. 
2. Select a target column by clicking on the list view item on the left side of the screen. Then select it with the select button below. 
3. Convert all the categorical columns with the convert button. This step is crucial if you don't do this step you can't train the algorithms. Because the machine doesn't understand strings, it only understands numbers. 
4. Remove all null values from the dataset. This step is also crucial.  
5. Perform data preprocessing techniques. For example, you can remove columns or select the best features. This step is optional.  
6. Move into the next tab. Here you can split the dataset using Hold-Out or K-Fold.  
7. In the classification tab, you can select an algorithm and train the algorithm. After the training process, the scores are will be printed below. Also, you can use buttons to view the confusion matrix, predicted and true values, and roc curve.  
8. Ensemble tab you can select an algorithm and train it. Also, you can view scores and graphs after the training process.  
9. Search tab is the same as the previous step you can select an algorithm and train it and view scores and graphs.


