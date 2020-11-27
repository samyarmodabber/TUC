import numpy as np
from scipy.stats import t

def multipleTTest(X, Y, includeIntercept = True, labels = []):
    
    # Get number of samples and number of coefficients
    n, p = X.shape
    
    if includeIntercept:
        # Stack one column containing only ones in front of X
        X = np.hstack((np.ones((n,1)),X))
        
    else:
        
        # Correct number of coefficients, remember intercept is \beta_0
        p = p - 1
    
    # Invert X^T * X
    V = np.linalg.inv((X.T).dot(X))
    
    # Compute regression coefficients beta
    beta = V.dot( X.T.dot(Y) )
    
    # Extract diagonal out of matrix (X^T * X)^-1
    v = V.diagonal()
    
    # Predict y using beta
    y_pred = X.dot(beta)
    
    # Compute residual sum of squares
    RSS = np.power(Y - y_pred,2).sum()
    
    # Compute estimate of sigma
    sigma_hat = np.sqrt( 1./(n-p-1) * RSS )
    
    # Compute the standard errors
    SE = np.sqrt(v) * sigma_hat
    
    # Compute the values of the t-statistic
    t_vals = beta / SE
    
    # Compute the corresponding p values
    p_vals = 2*t.cdf(-np.absolute(t_vals), n-p-1)
    
    # Print header
    print('|  Coefficient | Estimate |    SE    | t-statistic |  p-value  |')
    print('---------------------------------------------------------------')
    
    # Print 
    for i in range(p+1):
        pval = p_vals[i]
        if pval < 0.0001:
            pval_str = '< 0.0001'
        else:
            pval_str = '  %5.4f' % pval
            
        if len(labels) == 0:
            beta_str = 'beta_%02d' % i
        else:
            if includeIntercept:
                if i == 0:
                    beta_str = 'Intercept'
                else:
                    beta_str = labels[i-1]
            else:
                beta_str = labels[i]
        print('| %10s   |  %6.3f  |  %6.4f  |    %5.2f    | %s  |' \
              % (beta_str, beta[i], SE[i], t_vals[i], pval_str))
    
    return RSS