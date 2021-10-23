#!/usr/bin/env python
# coding: utf-8

# # Work Distribution

# In[18]:


import pandas as pd
import numpy
from pulp import *


# # Linear Programming Model

# In[19]:


prob=LpProblem("Minimize",LpMinimize)


# In[20]:


#time taken to finish each work
w1=LpVariable("w1",lowBound=1)
w2=LpVariable("w2",lowBound=1)
w3=LpVariable("w3",lowBound=1)
w4=LpVariable("w4",lowBound=1)
w5=LpVariable("w5",lowBound=1)
w6=LpVariable("w6",lowBound=1)
w7=LpVariable("w7",lowBound=1)
w8=LpVariable("w8",lowBound=1)
w9=LpVariable("w9",lowBound=1)
w10=LpVariable("w10",lowBound=1)
w11=LpVariable("w11",lowBound=1)
w12=LpVariable("w12",lowBound=1)
w13=LpVariable("w13",lowBound=1)
w14=LpVariable("w14",lowBound=1)
w15=LpVariable("w15",lowBound=1)


# In[21]:


#objective function is find minimum total time required to finish each work
prob+=(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15)


# # Contraints
# 1) Each machine must take some break if task is assigned to it consecutively

# # Solution

# ## Data CleanUp:

# In[22]:


def data(filename):
    df=pd.read_csv(filename)
    df=df.drop(columns=['Unnamed: 0'])
    df=df.rename(columns={"breaks between works": "breaks"}, errors="raise")
    breaks = df["breaks"].to_numpy()
    df=df.drop(columns=['breaks'])
    return df, breaks


# # Heuristic Approach to Solve Linear Programming:
#  This solution is built using few heurisitcs, the search space is explored to obtain the best possible answer.
# 
# Each column is converted into an array, each column represents timings for each work to a specific machine.
# The algorithm starts by assigning work in an ascending sequence, and at each step it is ensured that the timing selected is minimal from the list obtained in step given above and we try to minimize the break times by ensuring that the consecutive works are not allocated to the same machine.
# The result is stored in a dictionary, keys being the work id and the values are the machines that a work is assigned to.
# Once all the works in the problem statement are allocated to some machine, we have to consider the break times.
# Loop through the allocations dictionary to check whether two consecutive works are allocated to same machines.
#    

# In[23]:


#function to find allocation, the parameter passed is the pandas dataframe.
def find_allocation(filename):
    #dictionary to store the final allocation.
    allocations={}
    #runtime obtained from the allocations.
    total_time=0
    #list for search space.
    l=[]
    #a counter for operations.
    c=0
    df,breaks=data(filename)
    
    #loop through all the works to be allocated to machines.
    for i in df.columns:
        c=c+1
        #convert the timings to list
        values = df[i].to_numpy()
        #find the minimum timing
        result = numpy.where(values == numpy.amin(values))
        #add the minimum time to the result
        total_time += numpy.amin(values)
        #if the first or the last works from  the list
        if(c==1 or c==len(df.columns)):
            allocations[i] = ((result[0][0]+1))
        #if the work is b/w last and first
        if(l):
            #below is the heuristic to perform allocation
            find=l[len(l)-1]
            work_assigned=0
            for k in find:
                for item in result[0]:
                    if(k==item):
                        continue
                    else:
                        work_assigned=1
                        #work is allocated to the machine which is (item+1).
                        allocations[i] = ((item+1))
                        #append the result in l for further processing.
                        l.append(result[0])
        
        l.append(result[0])   
    
    #after the allocation is found, aim is to find the break times required and add to total time.

    counter=0
    previous_machine=None
    #loop throught the result.
    for i in allocations:
        #print(i)
        #check if allocation is first allocation.
        if(counter==0):
            counter+=1
            previous_machine=allocations[i]
            allocations[i]="Machine: "+str(allocations[i])
            continue
        #check if the allocation is similar to previous allocation, if yes, add the break time to total time.
        if(allocations[i]==previous_machine):
            break_time=breaks[allocations[i]]
            total_time+=break_time
        previous_machine=allocations[i]
        #edit allocation to make it more readable.
        allocations[i]="Machine: "+str(allocations[i])
        
    return allocations,total_time


# In[24]:


#call the method to obtain result, just pass the data through.
filename="WorkDistribution.csv"
allocations,total_time=find_allocation(filename)


# In[25]:


total_time


# In[26]:


allocations


# ## 2nd Approach

# This is the 2nd approach to the Work distribution question.
# 
# In this approach we consider the possibility that the different works/machines are not related to each other at any way possible and the state of one work/machine (either not started, implementing or completing) won't affect the other work/machine in any manner.
# 
# Considering no entity is affected by any other entity's state, we can run the machines in parallel and save a lot of time i.e., multiple machines can execute multiple work at a single time.
# 
# Below is the code for the approach of solving the problem so that all the work can be done in minimum time.

# ## Libraries

# This is a small section where the libraries which we want will be imported.
# 
# Below are the libraries which we need to solve this particular problem.

# In[27]:


import pandas as pd
import numpy as np
from collections import OrderedDict


# ## Problem Formulation

# In this section we will tailor the csv file and make some necessary changes which we will need to solve the problem using the algorithm designed.

# In[28]:


# In the below line, the csv file is converted into a dataframe using the pandas library
df=pd.read_csv("WorkDistribution.csv")


# In[29]:


# Here is the representation of the data distribution in the csv file provided in the form of dataframe
df


# In[30]:


# Below are the columns which are present in the dataframe
df.columns


# In[31]:


# For simplicity, we are converting the last column with the long name into a shorter one
df=df.rename(columns={"breaks between works": "breaks"}, errors="raise")

df


# In[32]:


# For considering the breaks between the works and to make the work easier here we are creating one more dataframe
# just for the break time of multiple machines
breaks = df[['Unnamed: 0','breaks']]


# In[33]:


# Below is the representation of the 'breaks' dataframe
breaks


# In[34]:


# as we don't need the breaks column in the main dataframe, we are removing the 'breaks' column from the main dataframe
df=df.drop(columns=['breaks'])


# In[35]:


# Below is the final representation of the dataset with which we will work with
df


# In[36]:


# For future calculations in the path to solving the problem, we need to convert the dataframe into a matrix
# Below is the code to convert the dataframe into a format of a matrix
df_matrix = df.to_numpy()


# In[37]:


df_matrix


# In[38]:


# As shown in the above matrix, the first column of the matrix is filled with the machine number, we don't need this for
# the formulation, so we will delete that column
df_matrix = np.delete(df_matrix, (0), axis = 1)


# ## Problem Solving

# This is the main section of the program.
# 
# This is the algorithm which is used in order to solve the problem, considering the machines and work can be implemented in parallel with no affect on each other's performance

# In[13]:


## Below are the creation of variables
## In these variables we will store the solution to the problem
## Work is the total runtime of one particular machine (including breaks).
## Allocations is the work allocation on the different machines
## Work and Allocations are defined as libraries here, i.e., it will contain a key value pair.
## In each of the variable key is the Machine which will be operation the work and value is the actual answer/purpose of the variable
work = OrderedDict()
allocations = {}

## This problem is designed in a way to implement the work for which a particular machine is fastest for
## We will be allocating work to the machines by finding out the work for which the machine is fastest for
## This is a different approach as we are not using the approach where we are finding the machine which can complete the work 
## in the fastest time.


## In this problem we will be using 2 different loops
## In the first loop we will be simply allocating the works to each machine
## We will be allocating that work for which a particular machine is fastest for.
## If the work is taken already we will drop that column and will find the fastest work for a particular machine out of the
## rest of the remaining works

## Below is the first loop
## We are running it for every machine i.e., implement the block of code for a single machine
for i in range(len(df['Unnamed: 0'])):
    ## The below if statement is present so that when the number of work is less than the number of machines
    ## then the loop should come to a break after allocating the last work
    if len(df.columns) == 1:
        break;
    ## Below we are filling up the work variable i.e., Making new key-value pair for each machine and the time associated with 
    ## the work. Key is the machine number and value is the work which can be completed by this machine the fastest
    work["Machine " + str(i+1)] = min(df_matrix[i])
    
    ## Below we are trying to find out the column number of the work and deleting that particular column  for work from 
    ## the matrix
    index = np.where(df_matrix[i] == min(df_matrix[i]))
    df_matrix = np.delete(df_matrix, (index[0][0]), axis = 1)
    index[0][0] = index[0][0] + 1
    ## In the below line , we are making the key-value pair for the allocations variable which will store the machine as key
    ## and the work completed by the machine as the value for the pair.
    allocations["Machine " + str(i+1)] = [(df.iloc[:,index[0][0]]).name]
    ## Below we are deleting the work column which is allocated to the machine
    df = df.drop(df.iloc[:,[index[0][0]]], axis = 1)

## This is the 2nd loop for the problem
## This is for the 2nd and more pair of work allocations for a single machine
## When the number of work is more than the number of number of machines, this loop will be implemented and allocate 2nd pair of
## works to every machine until every machine has 2 works allocated to them and it will continue for the next pair of works
## until every work is assigned to the machine
while len(df.columns)>=2:
    ## In this below line, we find out the minimum execution time of the system of machines i.e., we find out which machine
    ## will finish first after completing the allocated work
    key = min(work, key=work.get)
    ## After finding out the machine, we find out the index of that particular machine using the dictionary
    i = list(work.keys()).index(key)
    ## here we are finding out the work which can be completed by this machine machine the fastest and finding out the index 
    ## from the matrix given
    index = np.where(df_matrix[i] == min(df_matrix[i]))
    num = index[0][0]
    ## Here we are allocating the work to the machine and also adding up the time in the work dictionary.
    ## The new value for this particular key value pair is the total execution time of the machine including the break
    ## the value is equal to sum of the execution time of all the work and sum of total number of breaks taken in between 
    ##( total number of breaks = Total number of work executed by the machine - 1 )
    work[key] = int(work.get(key) + df_matrix[i][num] + breaks.iloc[(breaks[breaks['Unnamed: 0'] == key].index)[0]][1])
    ## Below we are adding the work into the allocations of the machine
    temp = allocations.get(key)
    temp_num = num + 1
    temp.append((df.iloc[:,temp_num]).name)
    ## Below we are deleting the assigned work columnf from the matrix and from the dataframe
    df_matrix = np.delete(df_matrix, (index[0][0]), axis = 1)
    index[0][0] = index[0][0] + 1
    df = df.drop(df.iloc[:,[index[0][0]]], axis = 1)


# In[14]:


## Final view of the dataframe showing every work has been assigned
df


# In[15]:


## Here are the different work allocations to different machines
allocations


# In[16]:


## Here is the total runtime of every machine including the break time took
work


# In[17]:


## below is the maximum runtime of the total system of machines
key = max(work, key=work.get)
value = work.get(key)
print("The total execution time of the system of machine is " + str(value) + " mins.")

