# BE_predictor
Using neural network to predict the binding energy.  
Using tool kit in MATLAB to build BP NN for the Binding energy  
[nn, error_re_ave, X_ps, Y_ps] = BE_predictor_v1_1(X_original, Y_original, LOO, breakpoints, range, ratio_train, units, coordinates, root)
  
  by wsyxbcl(wsyxbcl@gmail.com), 2017.3.12

## BE_predictor_v1.0
Add divide func: Change the way that data is divided, from continuous to
discrete.  
Change the calculation of ave_error to make it more reasonable.  
by yx_chai, 2017.3.18

## BE_predictor_v1.1
Change the way that data is divided, from 3 segments to 5, and more.  
Add two boolean input 'coordinates' and 'root', where coordinates refer
to the energy coordinate of input, and root refers to whether apply
sqrt() to the input data.  
by yx_chai, 2017.3.29

## BE_predictor_v1.1.1
Change the parameter of fitnet by network's property based in MATLAB.  
by yx_chai, 2017.3.30

## BE_predictor_v1.1.2
Add multielements feature, change the source of data to satisfy the need
of multielements training.  
Add LOO feature to the function, enable Leave-One-Out cross validation.  
And the ratio_train feature is kept to enable the test set, test set will
be replaced by loo_sample  
by yx_chai, 2017.4.1

## BE_predictor_v1.1.2.1
Bug fixed: in error_re_ave, put the abs to the Y_output step  
by yx_chai, 2017.4.1

## BE_predictor_v1.2(LOO version)
Add multielements feature, change the source of data to satisfy the need
of multielements training.  
Add LOO feature to the function, enable Leave-One-Out cross validation.  
And the ratio_train feature is kept to enable the test set, test set will
be replaced by loo_sample  
Bug fixed: in error_re_ave, put the abs to the Y_output step  
by yx_chai, 2017.4.1

Future plan
1. Try PCA to explore the density
2. Use noise to create more samples?
3. Inspired by the way they train imagenet(use random patches and reflection
of an image to get more data), maybe I can use different(but similar) slice of data
to get such job done.

A problem remained to be confirmed:  
Is mean normalization necessary since we remove the feature scaling part?  
Don't think that's important at this time, for when doing things with gradients, it
 works exactly the same, just take a record here.
