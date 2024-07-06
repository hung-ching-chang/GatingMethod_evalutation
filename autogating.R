# This script provides all autogating methods and theirs parameters
#### DeepCyTOF ####
# see DeepCyTOF_example.py
#### CyTOF Linear Classifier ####
setwd("/Users/auto-gating/reference/lda_ref")
source('CyTOF_LDAtrain.R')
source('CyTOF_LDApredict.R')
dt <-read.csv('data.csv',header = TRUE)  # data with cell type label (from manual gating)
# create training and testing (e.g., first 900 samples are training, and others are testing)
dt.Train <- dt[1:900,]
dt.Test <- dt[901:1000,]
write.table(dt.Train, file = 'data train/data_train.csv', col.names = FALSE, row.names = FALSE, sep = ',')
write.table(dt.Test, file = 'data test/data_test.csv', col.names = FALSE, row.names = FALSE, sep = ',')
LDA.Model <- CyTOF_LDAtrain(TrainingSamplesExt = 'data train/', TrainingLabelsExt = '', mode = 'CSV',
                            RelevantMarkers =  c(2:(ncol(AML.data)-1)),
                            LabelIndex = ncol(AML.data), Transformation = 'arcsinh')
Predictions <- CyTOF_LDApredict(LDA.Model, TestingSamplesExt = 'data test/',
                                mode = 'CSV', RejectionThreshold = 0)

#### ACDC (python) ####
# see ACDC_example.py
#### MP(Mondrian) ####
# see MP_example.py