#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# #  Investigate a Dataset - [noshowappointments-kagglev2-may-2016.csv]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# This dataset collects information
# from 100k medical appointments in
# Brazil and is focused on the question
# of whether or not patients show up
# for their appointment. A number of
# characteristics about the patient are
# included in each row.
# ● ‘ScheduledDay’ tells us on
# what day the patient set up their
# appointment.
# ● ‘Neighborhood’ indicates the
# location of the hospital.
# ● ‘Scholarship’ indicates
# whether or not the patient is
# enrolled in Brasilian welfare
# program Bolsa Família.
# ● Be careful about the encoding
# of the last column: it says ‘No’ if
# the patient showed up to their
# appointment, and ‘Yes’ if they
# did not show up.
# 
#  
# ### Dataset Description 
# 
#  We have csv file contain the data we are going to analyis it
# 
# 
# ### Question(s) for Analysis
# >**Tip**: Clearly state one or more questions that you plan on exploring over the course of the report. You will address these questions in the **data analysis** and **conclusion** sections. Try to build your report around the analysis of at least one dependent variable and three independent variables. If you're not sure what questions to ask, then make sure you familiarize yourself with the dataset, its variables and the dataset context for ideas of what to explore.
# 
# > **Tip**: Once you start coding, use NumPy arrays, Pandas Series, and DataFrames where appropriate rather than Python lists and dictionaries. Also, **use good coding practices**, such as, define and use functions to avoid repetitive code. Use appropriate comments within the code cells, explanation in the mark-down cells, and meaningful variable names. 

# In[45]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 
# 
# ### General Properties
# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.

# In[47]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
effect=pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
effect.head()


# In[48]:


effect.shape


# In[49]:


effect.info()


# There is No Null Values in Dataset

# In[50]:


effect.describe()


# In[51]:


effect.duplicated().sum()


# In[52]:


effect['PatientId'].duplicated().sum()


# i have 48228 patient have more than one show

# In[53]:


#check for the unique Patient
effect['PatientId'].nunique()


# i have only 62299 patient

# In[54]:


# the number of the unique values for each column
col = list(effect.columns.values)
for X in col:
    unique = effect[X].nunique()    
    print( "Number of unique values in {} is :".format(X) ,unique)


# In[55]:


# what is the unique values for some columns 
col = list(effect.columns.values)
for Y in col:
    if Y not in ['PatientId', 'AppointmentID','Age', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood']:
        unique_val = effect[Y].unique()    
        print( "Number of unique values in {} is :".format(Y) ,unique_val)


# 
# ### Data Cleaning
# After Discussing the Structure of the data and any problems then need to be cleaned , perform those cleaning Steps in Second part of this section

# In[56]:



effect.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)
effect.rename( columns={'No-show':'No_show'}, inplace=True)
effect.rename( columns={'Hipertension':'Hypertension'}, inplace=True)


effect.head()


# In[57]:


# Age looks weird
len(effect[effect["Age"] == 0]),len(effect[effect["Age"] < 0])


# In[58]:


#find patient whoes age=-1
wiredpatient=effect.query('Age==-1')
wiredpatient


# In[59]:


#drop this patientand remain Age=0 as akids
effect.drop(index=99832, inplace=True)


# now we drop this invalid row

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: we've trimmed and cleaned the data, we're ready to move on to exploration. Compute statistics and create visualizations with the goal of addressing the research questions that you posed in the Introduction section
# ### Generallook on Data

# In[60]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
effect.hist(figsize=(14,11));


# In[61]:


# Generating the correlation heatmap to compare the correlation between features
correlation = effect.corr()
sns.heatmap(correlation);


# there's no need to remove any feature

# In[62]:


#Draw pairplot kind scatter
sns.pairplot(effect, kind="scatter")
plt.show()


# In[63]:


# Number of patients at each age
plt.figure(figsize=(13,5))
plt.xticks(rotation=90)
xP = sns.countplot(x=effect.Age)
xP.set_title("Number of scheduled appointments according to each age")
plt.show()


# we can see that the most of appointmenst in age (0 and 1) and also alot in range (49 to 59)

# In[64]:


Show = effect.No_show== 'No'
NO_show =effect.No_show== 'Yes'


# In[65]:


effect[Show].count()


# In[66]:


effect[NO_show ].count()


# ## Investigate other factore affect on coming of patient

# # Research Question1

# What are the important factors for us to know if the patient will come on time?

# In[68]:


effect[Show].groupby(['Hypertension','Diabetes']).mean()['Age']
effect[NO_show].groupby(['Hypertension','Diabetes']).mean()['Age']


# In[69]:


#values count of All attendance
effect["No_show"].value_counts()


# In[70]:


#dos age of patient affect on attendance
def attendance (effect,col_name, attend, absent):
    plt.figure(figsize=[16,4]);
    effect[col_name][Show].hist(alpha=0.5,bins=15,color='blue',label='show')
    effect[col_name][NO_show].hist(alpha=0.5,bins=15,color='red',label='noshow')
    plt.legend();
    plt.xlabel('Age')
    plt.ylabel('PatientNumber');
attendance(effect,'Age', show, noshow)


   


# In[71]:


#represet the attendance percentage with pie chart
plt.title ("(%) Attendance vs. no-show on appointments", fontsize=14)
effect["No_show"].value_counts().plot(figsize=(5,5),kind="pie",autopct='%.2f', textprops={'fontsize': 14});


# In[72]:


#doesageand chornic diseases are affected on the attendance rat together
plt.figure(figsize=[16,4]);
effect[Show].groupby(['Hypertension','Diabetes']).mean()['Age'].plot(kind='bar',color='blue',label='show')

effect[NO_show].groupby(['Hypertension','Diabetes']).mean()['Age'].plot(kind='bar',color='red',label='noshow')

plt.legend()
plt.title('comparison acc toage, chronic diseases')
plt.xlabel('chronicdiseases')
plt.ylabel('MeanAge')


# In[73]:


# check the value count of all attendance
effect["Gender"].value_counts()


# In[74]:


#represet the attendance percentage with pie chart
plt.title ("(%) Attendance vs. noshow on appointments", fontsize=14)
effect["Gender"].value_counts().plot(figsize=(6,6),kind="pie",autopct='%.2f', textprops={'fontsize': 15});


# In[75]:


#does  age and Gender affect on the attendance together( check by meean age)
plt.figure(figsize=[12,4]);
effect[Show].groupby('Gender').Age.mean().plot(kind='bar',color='blue',label='show')
effect[NO_show].groupby('Gender').Age.mean().plot(kind='bar',color='red',label='noshow')
plt.legend()
plt.title('comparison acc to mean age , Gender')
plt.xlabel('Gender')
plt.ylabel('meanAge')


# In[78]:


#show the number of each gender versus the attendance by bar chart¶
Attends = effect['No_show']
show = sns.countplot(x=effect.Gender, hue=Attends)
show.set_title("the number of each gender versus the attendance")
x_ticks_labels=['F', 'M']
plt.show();


# We notice that the number of Female patients is higher in both showing and not-showing, compared to the male patients, which is understandable , due to higher number of appointments for the females

# In[79]:


#No-show percentage for female patients versus male patients
fig, axs = plt.subplots(1,2)
effect[effect["No_show"]=="No"].groupby("Gender")["No_show"].count().plot(figsize=(12,4),kind="pie",autopct='%.2f', ax=axs[0],colors=['green','yellow'],title="Show",textprops={'fontsize': 15});
effect[effect["No_show"]=="Yes"].groupby("Gender")["No_show"].count().plot(figsize=(12,4),kind="pie",autopct='%.2f',ax=axs[1],colors=['green','yellow'],title="No_show", textprops={'fontsize': 15});


# 64.56% of patients attending for their appointments are female, while males represent 35.44% of the total patients. The non-attendance rate was high, with 65.14% for females, compared to 34.86 for males, due to the high number of appointments booked by females.

# In[80]:


# check the count of appointments for different ages
effect["Age"].value_counts()


# In[81]:


#represet the attendance percentage with pie chart
plt.title ("(%) Attendance vs. no-show on appointments", fontsize=14)
effect["No_show"].value_counts().plot(figsize=(4,4),kind="pie",autopct='%.2f', textprops={'fontsize': 15});


# In[82]:


# Number of patients at each age
plt.figure(figsize=(13,5))
plt.xticks(rotation=90)
xP = sns.countplot(x=effect.Age)
xP.set_title("Number of scheduled appointments according to each age")
plt.show()


# we can see that the most of appointmenst in age (0 and 1) and also alot in range (49 to 59)

# In[84]:


#the precentage of sex influance
def attendance(effect, col_name,attend,absent):
    plt.figure(figsize=[12,4])
    effect[col_name][Show].value_counts(normalize=True).plot(kind='pie',label='show')
    plt.legend();
    plt.title('comparison between attendance by Gender')
    plt.xlabel("Gender")
    plt.ylabel('PatientNumber');
attendance(effect,'Gender',show,noshow)


# In[86]:


#does receving SMS  affect attendance
def attend (effect,colname,attend, absent):
    plt.figure(figsize=[10,3]);
    effect[colname][Show].hist(alpha =0.5,bins=15, color='green',label='Show')
    effect[colname][NO_show].hist(alpha =0.5, bins=15 ,color='red', label='noshow')
    plt.legend();
    plt.title('Comparison of acc to sms receiver')
    plt.xlabel('SMS')
    plt.ylabel('PatientNumber');
attend(effect,'SMS_received', Show, NO_show)


# In[87]:


#represet the SMS_received percentage with pie chart
plt.title ("(%) of Patients who received the SMS vs the who didn't receive the SMS", fontsize=15)
effect["SMS_received"].value_counts().plot(figsize=(6,6),kind="pie",autopct='%.2f', textprops={'fontsize': 15});


# In[92]:


# the precentage of the patients that recieved and show or no show up vs they don't
# the precentage of the patients that recieved and show or no show up vs they don't
fig, axs = plt.subplots(1,2)
effect[effect["SMS_received"]==1].groupby("No_show")["SMS_received"].count().plot(figsize=(10,5),kind="pie",autopct='%.2f', ax=axs[0],colors=['lightblue','orange'],title="SMS_recieved",textprops={'fontsize': 14});
effect[effect["SMS_received"]==0].groupby("No_show")["SMS_received"].count().plot(figsize=(10,5),kind="pie",autopct='%.2f',ax=axs[1],colors=['lightblue','orange'],title="no_SMS_recieved",textprops={'fontsize': 14});


# We can see that the percentage of patients who did not receive an SMS and show the appointment is greater than what they did not receive by 12.97% because the number of patients who did not receive was greater as we saw before
# 
# But I can see that the percentage of patients showing though they did not receive SMS is closer than the percentage that patients received despite the big difference between their accounts, so I can imagine that receiving SMS will affect attendance significantly

# In[94]:


#deos the Handcap affect the attendance
def attendance (effect ,colname, attend,absent):
    plt.figure(figsize=[12,4]);
    effect[colname][Show].hist(alpha=0.5,bins=15,color='blue',label='show')
    effect[colname][NO_show].hist(alpha=0.5,bins=15,color='red',label='noshow')
    plt.legend();
    plt.title('comparison acc to Handcap ')
    plt.xlabel('Handcap')
    plt.ylabel('Patientnumber');
attendance(effect,'Handcap',show,noshow)


# In[95]:


#dose Scholarship affect on attendance
def attendance (effect ,colname, attend,absent):
    plt.figure(figsize=[12,4]);
    effect[colname][Show].hist(alpha=0.5,bins=15,color='blue',label='show')
    effect[colname][NO_show].hist(alpha=0.5,bins=15,color='red',label='noshow')
    plt.legend();
    plt.title('comparison acc to Scholarship  ')
    plt.xlabel('Scholarship')
    plt.ylabel('Patientnumber');
attendance(effect,'Scholarship',show,noshow)


# <a id='conclusions'></a>
# ## Conclusions
#  the Age has its role as thosein the 0-10age group were the most show up .followed by the age group 36-70
#  
#  number of showing patient without receiving SMS is greater than number of showing with receiving SMS
#  
#  precentage of female whoes attend is greater than the precentage of male who 
#  
#  The total number of patients who attended is more than those who did not attend, and the number of patients who did not attend was 20.19%
# 
# Females tend to make appointments more than males, but males are more committed to attending
# 
# The younger the patient, the more appointments are made, although we note that patients between 50 and 80 years of age are more committed to attending appointments
# 
# Patients who do not have a scholarship are more committed to attending appointments than patients who have a scholarship
# 
# 
# 
# 
# > **Tip**: If you haven't done any statistical tests, do not imply any statistical conclusions. And make sure you avoid implying causation from correlation!
# 
# ### Limitations
# I could not detect the correlation  between patient showing, noshowing and many charactristics such as gender ,chornic disease,
# diabetes
# There are a few wrong data need to be explated, negative age values ​​and wrong scheduling dates
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[96]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




