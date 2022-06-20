## autoMFA 1.0.0
* This is the initial release version of autoMFA

## autoMFA 1.1.0
* This is the first update for autoMFA
* Made the returned objects instances of the "MFA" class
* Added print, plot and summary functions for the MFA class
* Changed the name of method `AMFA.inc' to 'AMFA_inc', added a depreciation warning to AMFA.inc
* Added the data to the diagnostics object in the model output
* Added the original call to the diagnostics object in the model output
* Changed the maxtries input from the vbmfa method to be numTries, making it consistent with the input of AMFA_inc which has the same purpose
* Changed the data input from the amofa method to be Y, making it consistent with the rest of the package
* Removed an unnecessary line from the test data set example.
* Changed the example dataset, updated its name to testDataMFA and improved its documentation. 