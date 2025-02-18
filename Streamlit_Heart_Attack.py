#Streamlit Heart Attack Prediction Project
#PROTOTYPE 1
#Developed By: JASHWANTH RAJ J.R

import pandas as pd
import streamlit as st

df=pd.read_csv('/heart_failure_clinical_records_dataset.csv')

#Segregate dataset into input X and Output Y
x1=df.iloc[:,:-1].values
y1=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x1,y1,test_size=0.35,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Training
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(X_train,Y_train)


# Predicting Whether You Die Or Not
# (age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,
#   75,      0,                     582,       0,               20,
#  high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time,DEATH_EVENT)
#                    1,   265000,             1.9,         130,  1,      0,   4,        1
st.title(":skull_and_crossbones:\tHEART ATTACK \tDEATH PREDICTION\t:coffin:")
st.header("*Based On Heart Failure Clinical Records Dataset\t:books:")
st.sidebar.title("HEART ATTACK DEATH PREDICTION")
st.sidebar.header('-->Follow The Example For Reference<--')

#SideBar
st.sidebar.subheader('-->:man-woman-girl-boy:\tAGE:\n It Should Be Greater Than 40') #In Header \n Works not in write function

st.sidebar.subheader('-->:wind_blowing_face:\tANAEMIA:\n It appears that anaemia is a condition characterized by a lower-than-normal level of red blood cells or haemoglobin in the blood.\nMore Info:Types of Anaemia:\nIron deficiency anaemia: caused by a lack of iron in the body, leading to a reduction in the number of red blood cells.\nAplastic anaemia: a condition where the bone marrow fails to produce enough red blood cells, white blood cells, and platelets.\nSickle cell anaemia:a genetic disorder that affects the shape of red blood cells, leading to haemolysis and haemolytic anaemia.')

st.sidebar.subheader("-->:anger:\tCREATININE PHOSPHOKINASE:\n Creatinine phosphokinase (CPK) is an enzyme that is found in the heart, brain, and skeletal muscles. It helps to convert creatine into phosphocreatine, which is a high-energy molecule that is used to generate energy in the body.\n Basic healthy level of CPK in humans\nThe normal range of CPK levels in the blood can vary depending on the laboratory and the individual, but generally,\n it falls between 10 to 120 micrograms per liter (mcg/l).")
st.sidebar.write("Here's a breakdown of the normal CPK ranges:")
st.sidebar.write("*Normal range: 10-120 mcg/l")
st.sidebar.write("*Males: 30-120 mcg/l")
st.sidebar.write("*Females: 10-80 mcg/l")

st.sidebar.subheader("-->:doughnut:\tDIABETES[SUGAR]")
st.sidebar.write("0.No")
st.sidebar.write("1.Yes")

st.sidebar.subheader("-->:heartbeat:\tEJECTION FRACTION")
st.sidebar.write("Ejection fraction (EF) is a measurement of how well the heart pumps blood out to the body with each heartbeat. It is expressed as a percentage, with a normal EF range of 50-70%. EF is an important indicator of heart health, and a low EF can be a sign of heart failure.")
st.sidebar.write("Here's a breakdown of the normal EF ranges:")
st.sidebar.write("*Normal EF: 50-70%")
st.sidebar.write("*Mildly abnormal EF: 41-49%")
st.sidebar.write("*Moderately abnormal EF: 30-40%")
st.sidebar.write("*Severely abnormal EF: below 30%")

st.sidebar.subheader("-->:drop_of_blood:\tHIGH BLOOD PRESSURE")
st.sidebar.write("0.No")
st.sidebar.write("1.Yes")

st.sidebar.subheader("-->:petri_dish:\tPLATELETS")
st.sidebar.write("Platelets, also known as thrombocytes, are small, irregularly-shaped blood cells that play a crucial role in blood clotting. They are produced in the bone marrow and circulate in the blood, where they can be activated to form a platelet plug to stop bleeding when a blood vessel is injured.")
st.sidebar.write("The normal range of platelet count is typically between 150,000 to 450,000 platelets per microliter (μL) of blood.")
st.sidebar.write("Here's a breakdown of the normal platelet count ranges:")
st.sidebar.write("*Normal platelet count: 150,000-450,000/μL")
st.sidebar.write("*Thrombocytopenia (low platelet count): below 150,000/μL")
st.sidebar.write("*Thrombocytosis (high platelet count): above 450,000/μL")

st.sidebar.subheader("-->:droplet:\tSERUM CREATININE")
st.sidebar.write("Serum creatinine is a waste product that is produced by the normal breakdown of muscle tissue in the body. It is removed from the blood by the kidneys and excreted in the urine. Serum creatinine levels are often used as a measure of kidney function, as high levels can indicate kidney damage or disease.")
st.sidebar.write("The normal range of serum creatinine is typically between 0.6 to 1.2 milligrams per deciliter (mg/dL) for adults.")
st.sidebar.write("Normal serum creatinine: 0.6-1.2 mg/dL")
st.sidebar.write("*Mildly elevated serum creatinine: 1.2-1.5 mg/dL")
st.sidebar.write("*Moderately elevated serum creatinine: 1.5-2.0 mg/dL")
st.sidebar.write("*Severely elevated serum creatinine: above 2.0 mg/dL")

st.sidebar.subheader("-->:battery:\t:bulb:\tSERUM SODIUM")
st.sidebar.write("Serum sodium is an electrolyte that is essential for maintaining proper fluid balance and nerve and muscle function in the body. It is measured in the blood as part of a routine electrolyte panel.")
st.sidebar.write("The normal range of serum sodium is typically between 135 to 145 milliequivalents per liter (mEq/L).")
st.sidebar.write("Normal serum sodium: 135-145 mEq/L")
st.sidebar.write("Hyponatremia (low sodium): below 135 mEq/L")
st.sidebar.write("Hypernatremia (high sodium): above 145 mEq/L")

st.sidebar.subheader("-->:male_sign:\t:female_sign:\tGENDER")
st.sidebar.write("0.Female")
st.sidebar.write("1.Male")

st.sidebar.subheader("-->:smoking:\t:no_smoking:\tSMOKING")
st.sidebar.write("0.No")
st.sidebar.write("1.Yes")

st.sidebar.subheader("-->:running:\t:bed:\tHealth Condition[Time]")
st.sidebar.write("It is used to track the progression of heart failure in patients.")
st.sidebar.write("In Dataset, It Hold:")
st.sidebar.write("Minimum Value is: 4 Days")
st.sidebar.write("Maximum Value is: 280 Days")

st.sidebar.header("	:male-technologist:\tDeveloped By\t:male-technologist: ")
st.sidebar.header(":zap:\tJASHWANTH RAJ J.R")
st.sidebar.write('*Prototype Project With 80% Accuracy & Precision Score')
st.sidebar.write('*Entertainment Purpose Only')

#Input

st.header("	:pencil:")
Name=st.text_input("Enter Your Name:")

st.header(":man-woman-girl-boy:")
Age=st.number_input("\nEnter The Age:")

st.header(":wind_blowing_face:")
Anaemia=st.number_input("Do You Have Anaemia[0 or 1]:")

st.header(":anger:")
creatinine_phosphokinase=st.number_input("Enter Your Creatinine Phosphokinase[1 to 2413]:")

st.header(":doughnut:")
diabetes=st.number_input("Do You Have Diabetes[No=0 or Yes=1]:")

st.header(":heartbeat:")
ejection_fraction=st.number_input("Enter Ejection Fraction Value[1 to 100]:")

st.header(":drop_of_blood:")
high_blood_pressure=st.number_input("Do You Have High Blood Pressure[0 or 1]:")

st.header("	:petri_dish:")
platelets=st.number_input("Enter Platelets[150,000-450,000] Number:")

st.header("	:droplet:")
serum_creatinine=st.number_input("Enter The Serum_Creatinine Level[1.0 to 10.0]:")

st.header(":battery:\t:bulb:")
serum_sodium=st.number_input("Enter The Serum_Sodium Level[1 to 150]:")

st.header(":male_sign:\t:female_sign:")
sex=st.number_input("Enter Your Gender [ Male=1, Female=0 ]:")

st.header(":smoking:\t:no_smoking:")
smoking=st.number_input("Do You Smoke[Yes=1, No=0]:")

st.header(":running:\t	:bed:")
time=st.number_input("Enter Health Level:[Strong>1 to 285<Weak]:")

new=[[Age,Anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]]

result=model.predict(sc.transform(new))

if result==1:
    st.success(f":smiley:\t {Name} You Are Healthy:smile:")
else:
    st.error(f":joy_cat:\t {Name} You Will Die Soon\t:bell:\t[REST IN PEACE!!!]")

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
cm=confusion_matrix(Y_test,y_pred)

st.info(":chart_with_upwards_trend:\tACCURACY OF THE MODEL:{0}%".format(accuracy_score(Y_test,y_pred)*100))
st.info(":bar_chart:\tPRECISION OF THE MODEL:{0}%".format(precision_score(Y_test,y_pred)*100))
st.info(":dizzy:\tRECALL OF THE MODEL:{0}%".format(recall_score(Y_test,y_pred)*100))
st.info(":speech_balloon:\tF1 OF THE MODEL:{0}%".format(f1_score(Y_test,y_pred)*100))
st.write('*Entertainment Purpose Only')
