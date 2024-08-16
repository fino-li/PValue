import numpy as np
import pandas as pd
from scipy import stats

df=pd.read_csv('data.csv')
distf=pd.read_csv('distribution.csv', header=None)

data = df.iloc[:, 0].values
distribution = distf.values

PValue = []

def interpolate_p_value(A2, critVals, sigLevel):
    for i in range(len(critVals)-1):
        if critVals[i] <= A2 <= critVals[i+1]:
            slope = (sigLevel[i+1]-sigLevel[i]) / (critVals[i + 1] - critVals[i])
            intercept = sigLevel[i] - slope * critVals[i]
            return slope * A2 + intercept
    if A2 < critVals[0]:
        return sigLevel[0]
    # If A2_stat is greater than the largest critical value
    if A2 > critVals[-1]:
        return sigLevel[-1]

#normal dist
result = stats.anderson(data, dist='norm')
p_value = interpolate_p_value(result.statistic, distribution[1], distribution[0])
PValue.append(p_value)

#Log Normal wrong
log_data = np.log(data)
result = stats.anderson(log_data, dist='norm')
p_value = interpolate_p_value(result.statistic, distribution[3], distribution[2])
PValue.append(p_value)

#Exponential wrong
result = stats.anderson(data, dist='expon')
p_value = interpolate_p_value(result.statistic, distribution[5], distribution[4])
PValue.append(p_value)


#Logistic wrong
result = stats.anderson(data, dist='logistic')
p_value = interpolate_p_value(result.statistic, distribution[7], distribution[6])
PValue.append(p_value)


#Largest Extreme wrong
result = stats.anderson(data, dist='gumbel')
p_value = interpolate_p_value(result.statistic, distribution[9], distribution[8])
PValue.append(p_value)


#Smallest Extreme wrong
result = stats.kstest(data, 'weibull_min', args=(stats.weibull_min.fit(data)))
PValue.append(result.pvalue)


#Weibull
ks_statistic, p_value = stats.kstest(data, 'weibull_min', args=(stats.weibull_min.fit(data)))
PValue.append(p_value)

#Gamma
ks_statistic, p_value = stats.kstest(data, 'gamma', args=(stats.gamma.fit(data, floc=0)))
PValue.append(p_value)

print(PValue)

