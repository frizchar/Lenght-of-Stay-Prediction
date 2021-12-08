# import dependencies
import pandas as pd
from sklearn import linear_model

# select predictor columns to read from initial dataset
col_list_X = ["ΗΛΙΚΙΑΚΗ_ΟΜΑΔΑ","ΦΥΛΟ_ΑΣΘΕΝΗ","ΟΙΚΟΓΕΝΕΙΑΚΗ_ΚΑΤΑΣΤΑΣΗ","ΚΑΤΗΓΟΡΙΑ_ΕΠΑΓΓΕΛΜΑΤΟΣ_ΑΣΘΕΝΗ","ΚΩΔ_ΘΕΡΑΠΟΝΤΟΣ_ΙΑΤΡΟΥ","ΚΩΔ_ΟΜ_ICD10_ΕΙΣΙΤ",
			  "ΚΑΤΗΓΟΡΙΑ_ΝΟΜΟΥ_ΑΣΘΕΝΗ","ΚΑΤΗΓΟΡΙΑ_ΕΘΝΙΚΟΤΗΤΑΣ_ΑΣΘΕΝΗ","ΜΗΝΑΣ_ΕΙΣΑΓΩΓΗΣ","ΠΡΩΙΝΗ_ΕΙΣΑΓΩΓΗ","ΗΜΕΡΑ_ΒΔΟΜΑΔΑΣ_ΕΙΣΑΓΩΓΗΣ",
			  "SUM_PATASFAL","ΘΡΗΣΚΕΥΜΑ_ΑΣΘΕΝΗ"]
# select target column to read from initial dataset
col_list_y = ["LOS_GROUP"]											
# add target column to predictor columns
col_list = col_list_X + col_list_y

# read dataset
df = pd.read_csv("data/los_pred_feed.csv", usecols=col_list)		#export from sql developer with UTF-8 encoding

df = df.dropna() 													#drop null values

print("dataset shape : ",df.shape,"\n")
print("νοσοκομείο : ",pd.read_csv("data/los_pred_feed.csv")["ΝΟΣΟΚΟΜΕΙΟ"].unique(),"\n")
print("έτος έναρξης : ",pd.read_csv("data/los_pred_feed.csv")["ΕΤΟΣ_ΕΙΣΑΓΩΓΗΣ"].min(),"\n")

X = df[col_list_X] 
y = df[col_list_y] 

print("predictors 'X' before encoding : \n:",X.head(),"\n")
print("target 'y' before encoding : \n",y.head(),"\n")
print("target 'y' values : ",pd.read_csv("data/los_pred_feed.csv")["LOS_GROUP"].unique(),"\n")


#-------------- analyze target variable ------------#
# define target variable classes
class_0 = y[y['LOS_GROUP'] == '1 DAY (ODC PATIENTS)'] 
class_1 = y[y['LOS_GROUP'] == '2 ΤΟ 3 DAYS'] 
class_2 = y[y['LOS_GROUP'] == '4+ DAYS'] 

print("shapes of target 'y' classes :")

print('class 0:', class_0.shape) # print the shape of the class
print('class 1:', class_1.shape) 
print('class 2:', class_2.shape) 


#-------------- encode features ------------#
# import dependencies
from sklearn import preprocessing

# create labelEncoder
le = preprocessing.LabelEncoder()

# convert string labels into numbers
for xx in col_list:
  df[xx]=le.fit_transform(df[xx])
   
X = df[col_list_X] 
y = df[col_list_y] 

print("predictors 'X' after encoding : \n:",X.head(),"\n")
print("target 'y' after encoding : \n",y.head(),"\n")




#------------- implement random under-sampling with imblearn-----------#
# import dependencies
from imblearn.under_sampling import RandomUnderSampler

# fit predictor and target variable
rus = RandomUnderSampler(random_state=42, replacement=True)
X_rus, y_rus = rus.fit_resample(X, y)

print('original dataset shape :', y.shape)
print('resampled dataset shape :', y_rus.shape,'\n')
print('resampled dataset values\' count :\n'); print(y_rus.LOS_GROUP.value_counts(),'\n')


#-------------- train/test split ------------#
# import dependencies
from sklearn.model_selection import train_test_split

# split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.2, random_state=42)  # z% training and (100-z)% test on resampled data

					
#-------------- build model ------------#			
# import dependencies
from sklearn.ensemble import GradientBoostingClassifier

# build gradient boosting classifier model
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.08, loss='deviance', verbose=1,
								min_samples_leaf=5, min_samples_split=10,random_state=123,subsample=0.5)

param = gb.get_params() 
print("gradient boosting model parameters : \n", param, "\n")  #print all model parameters

# train the model using the training sets
gb.fit(X_train, y_train.values.ravel())

# predict the response for test dataset
y_pred = gb.predict(X_test)
								


#--------- evaluate model on test data -------#										
# import dependencies
from sklearn import metrics

# accuracy KPI
print("accuracy :                       ",format(metrics.accuracy_score(y_test, y_pred),'.2f'),'\n')

# precision KPI
print("precision w/ average = \'micro\' : ",format(metrics.precision_score(y_test, y_pred, average='micro'),'.2f'),'\n')

# recall KPI
print("recall w/ average = \'macro\' :    ",format(metrics.recall_score(y_test, y_pred, average='macro'),'.2f'))	
print("recall w/ average = \'macro\' :    ",format(metrics.recall_score(y_test, y_pred, average='macro'),'.2f'),'\n')	

# import dependencies
from sklearn.metrics import balanced_accuracy_score

# balanced accuracy KPI
print("balanced accuracy :              ",format(balanced_accuracy_score(y_test, y_pred),'.2f'))

# adjusted balanced accuracy KPI
print("balanced accuracy (adjusted) :   ",format(balanced_accuracy_score(y_test, y_pred,adjusted=True),'.2f'),'\n')

print('|----------------------------------|\n')



#------ generate validation dataset ---------#										

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=43)  # z% training and (100-z)% test on resampled data					

					
#Train the model using the training sets
gb.fit(X_train, y_train.values.ravel())

#Predict the response for test dataset
y_pred = gb.predict(X_val)


#--- evaluate model on validation data -------#										

# accuracy KPI
print("accuracy :                       ",format(metrics.accuracy_score(y_val, y_pred),'.2f'),'\n')

# precision KPI
print("precision w/ average = \'micro\' : ",format(metrics.precision_score(y_val, y_pred, average='micro'),'.2f'),'\n')

# recall KPI
print("recall w/ average = \'macro\' :    ",format(metrics.recall_score(y_val, y_pred, average='macro'),'.2f'),'\n')	

# balanced accuracy KPI
print("balanced accuracy :              ",format(balanced_accuracy_score(y_val, y_pred),'.2f'))

# adjusted balanced accuracy KPI
print("balanced accuracy (adjusted) :   ",format(balanced_accuracy_score(y_val, y_pred,adjusted=True),'.2f'),'\n')