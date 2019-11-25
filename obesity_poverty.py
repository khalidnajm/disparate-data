import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Preparing data
# =============================================================================

#read in files and save as DataFrames. ob = obesity, pov = poverty
ob = pd.read_excel('OB_PREV_ALL_STATES.xlsx',header=1,index_col='FIPS Codes',na_values='No Data')
pov = pd.read_csv('ACS_13_1YR_S1701_with_ann.csv',header=0,skiprows=[1],quotechar='"',na_values=['N','(X)'],index_col='GEO.id2')

#extract from pov only the FIPS and Percent poverty cols
pov = pov[['HC03_EST_VC01']]

#remove NAN values
ob = ob.dropna()

# =============================================================================
# EDA
# =============================================================================

#EDA on the Obesity data for 2004: number of persons who are obese
_ = plt.hist(ob['percent'], label = '2004')
_ = plt.title('2004')
_ = plt.xlabel('Percent of Obesity in County')
_ = plt.ylabel('Counts')

#show the plot
plt.show()

#initialise dictionary to store the means and standard devs
ob_vals = np.empty([10,2])

#plotting histograms for all years: 2004 to 2013
for i in range(1,10):
    #create 3x3 subplot for nine histograms excluding the first 'percent' column
    plt.subplot(3,3,i)
    
    #plot the histogram
    _ = plt.hist(ob['percent.'+str(i)], label = str(2003 + i))
    
    #give each subplot a title
    _ = plt.title(str(2003 + i))
    
    #append the mean and std to array ob_vals
    ob_vals[i][0] = np.mean(ob['percent.'+str(i)])
    ob_vals[i][1] = np.std(ob['percent.'+str(i)])
    
    #limit the axes
    axes = plt.gca()
    axes.set_xlim([10,50])
    axes.set_ylim([0,1200])
    
    #drop axis tick labels
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    
    
#give figure axes labels
_ = plt.xlabel('Percent of Obesity in County')

#tidy up the axes
_ = plt.tight_layout()

#show the plot
plt.show()

# =============================================================================
# Saving summary statistics
# =============================================================================

#convert list of means and stds to DataFrame
ob_dict = pd.DataFrame(ob_vals, columns = ['Mean','Std'])

#create a list of years from 2004 to 2014 inclusive
year = np.arange(2004,2014,1)

ob_dict['Year'] = year

#fill first row with percent data from 2004, which wasn't included in the for loop above
ob_dict.iloc[0,:] = (np.mean(ob['percent']),np.std(ob['percent']),2004)

#save the summary statistics above
ob_dict.to_csv('Obesity_Mean_Std')

#subset the DataFrame ob only with data of interest, country FIPS code already included
ob = ob[['percent.9']]

# =============================================================================
# Analysis: Finding the correlation coefficent between the prevalence of obesity and poverty
# =============================================================================

#first let's merge the two DataFrames
ob_pov = ob.merge(pov, left_index = True, right_index = True)

#rename columns
ob_pov.columns = ['Obesity (%)','Poverty (%)']

#rename index of merged DataFrame
ob_pov.index.name = 'Zipcodes'

#time to plot
sns.set()
_ = sns.lmplot('Obesity (%)','Poverty (%)',data = ob_pov, markers = '.')

#label the axes
_ = plt.xlabel('Obesity Prevalence (%)')
_ = plt.ylabel('Poverty Prevalence (%)')

#calculate the pearson corr coeff
ob_pov_corr, ob_pov_corr_pvalue = stats.pearsonr(ob_pov['Obesity (%)'], ob_pov['Poverty (%)'])

#annotate the plot
_ = plt.annotate(('$\\rho$ = ' + str(round(ob_pov_corr,2)) + ', p = ' + str('{:.2e}'.format(ob_pov_corr_pvalue))), 
                 xy = (15,40))

#show the plot
plt.show()
