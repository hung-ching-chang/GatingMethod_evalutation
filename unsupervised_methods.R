# This script provides all unsupervised methods and theirs parameters

library(flowCore) # for loading fcs files
dt <- read.FCS("HC_frozen_Tcell+01_1_CD45_.fcs", transformation = FALSE) # loading fcs files
markers <- read.table("cell_type_markers.txt") # markers for cell types

# preprocessing
dt <- dt[,markers] # Extract markers (for different cell types)
# asinh transformation to normalize the data
asinhTrans <- arcsinhTransform(a = 0, b = 0.2)
translist <- transformList(markers, asinhTrans)
dt.transform <- transform(dt, translist)
dt.transform <- exprs(dt.transform)

##### unsupervised clustering #####
#### ACCENSE (GUI) ####
write.FCS(dt, "ACCENSE/inputdata.fcs") # run on the ACCENSE app
#### CCAST ####
library(CCAST)
n.0 <- sum(dt.transform == 0)
min.mat <- min(dt.transform[dt.transform != 0])
dt.transform[dt.transform == 0] <- runif(n.0, min = min.mat/2, max = min.mat)
colnames(dt.transform) <- paste0("marker", 1:ncol(dt.transform))
colid <- 1:ncol(dt.transform)
marker.cv <- apply(dt.transform, 2, function(x){mean(x)/sd(x)})
marker1 <- names(marker.cv)[order(marker.cv, decreasing = TRUE)][1]
ccast.out <- ccast_main(file = dt.transform, ylabel = marker1, k = 20, 
                        transformlogic = FALSE, boot = NULL, deterministic = TRUE,
                        colid, coln = NULL, rown = NULL, npmix = FALSE, asinhp = 5, 
                        origscale = FALSE, groups = NULL, runsilhouette = FALSE)
#### ClusterX ####
library(cytofkit)
ClusterX.out <- ClusterX(dt.transform, dimReduction = NULL, 
                         outDim = 2,  gaussian = TRUE, 
                         alpha = 0.001, detectHalos = FALSE, 
                         SVMhalos = FALSE, parallel = TRUE, nCore = 20)
#### CosTaL (python) ####
# see CosTaL_example.py
#### Cytometree ####
library(cytometree)
CytomeTree.out <- CytomeTree(dt.transform)
#### densityCUT ####
library(densitycut)
densitycut.out <- DensityCut(dt.transform)
#### DensVM ####
library(cytofkit)
dt.transform.tsne <- cytof_dimReduction(dt.transform, method = "tsne")
DensVM.out <- DensVM(xdata = dt.transform, ydata = dt.transform.tsne)
#### DEPECHE ####
library(DepecheR)
depeche.out <- depeche(dt.transform, k = 20)
#### FLOCK (GUI) ####
write.table(dt.transform, "FLOCK/input.flowtxt", row.names = FALSE, col.names = TRUE)
# To compile the flock2 executable, use the following:
cc -o flock2 flock2.c -lm 
flock2 input.flowtxt
#### flowClust #### 
library(flowClust)
flowClust.out <- flowClust(dt, trans = 1, K = 20, min.count = -1, max.count = -1)

#### FlowGrid ####
write.csv(dt.transform, "flowgrid/FlowGrid-master/dt_tansform.csv", row.names = FALSE)
# Unix Command
pip3 install sklearn numpy scipy --user
cd flowgrid/FlowGrid-master
python3 sample_code.py --f dt_tansform.csv --n 4  --eps 1.1 --o FlowGrid.out.csv

#### flowMeans #### 
library(flowMeans)
flowMeans.out <- flowMeans(dt.transform, NumC = 20)

#### flowPeaks ####
library(flowPeaks)
flowPeaks.out <- flowPeaks(dt.transform)
#### FlowSOM ####
library(FlowSOM) 
FlowSOM.out <- FlowSOM(dt.transform, colsToUse = markers, nClus = 20)

#### immunoClust #### 
library(immunoClust)
immunoClust.out <- cell.MajorIterationLoop(dt.transform)

#### kmeans ####
kmeans.out <- kmeans(dt.transform, centers = 20)
#### PAC-MAN #### 
library(PAC)
PAC.out <- PAC(dt.transform, K = 20)

#### PhenoGraph #### 
library(cytofkit)
phenograph.out <- Rphenograph(dt.transform, k = 30)

#### Rclusterpp ####
library(Rclusterpp)
res <- Rclusterpp.hclust(dt.transform)
Rclusterpp.out <- cutree(res, k =  20)
#### SamSPECTRAL #### 
library(SamSPECTRAL)
SamSPECTRAL.out <- SamSPECTRAL(dt.transform,
                               k.for_kmeans = 20,
                               number.of.clusters = 20,
                               normal.sigma = 150,
                               separation.factor = 0.39)
#### SPADE #### 
write.FCS(dt, filename = "seletced_feature.fcs")
library(spade)
PANELS <- list(list(panel_files = c("seletced_feature.fcs"), 
                    median_cols = NULL, 
                    reference_files = c("seletced_feature.fcs"),
                    fold_cols = c()))
spade.out <- SPADE.driver("seletced_feature.fcs", out_dir = "output", 
                          cluster_cols = markers, panels = PANELS, 
                          transforms = flowCore::arcsinhTransform(a = 0, b = 0.2), 
                          layout = igraph::layout.kamada.kawai, 
                          downsampling_target_percent = 0.1, downsampling_target_number = NULL, 
                          downsampling_target_pctile = NULL, downsampling_exclude_pctile = 0.01, 
                          k = 50, clustering_samples = 50000)

#### SWIFT (GUI) ####
prepare fcs file and follow the tutorial: https://www.youtube.com/watch?v=8AG1ZworCm4

#### X-shift (GUI) ####
prepare fcs file and follow the tutorial: https://github.com/nolanlab/vortex/wiki/Getting-Started

