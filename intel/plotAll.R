library(readr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(reshape2)
library(plyr)
library(stats)
library(scales)
source("helpers.R")

smooth <- function(df,cols,width=2){
  df
}


plot3d <- function(df,axis,column,maintitle){
  dims = unique(df[[axis[1]]])
  #print(length(dims))
  print(dims)
  df[[axis[2]]] <- round(df[[axis[2]]],digits = 1)
  sparsities <-unique(df[[axis[2]]])
  
  print(sparsities)

  s <- split(df,df['Dim'])
  #print(length(dims))

  for(i in 1:length(dims)){
    #print(s[[i]])
  }
  
  map <-data.frame(
    lapply(1:(length(dims)),
      function(i){
        s[[i]][[column]]
      })
    )

   colnames(map) <- dims
   rownames(map) <- sparsities

   
   
  f=melt(df,id.vars=c("Dim","Sparsity"),measure.vars=paste(column))
  #print(f)
  p <- ggplot(data=f, aes(x=Dim, y=Sparsity, fill=value) ) + geom_tile() + scale_fill_gradient(low = "grey", high = "black") + ggtitle(maintitle) 
  return(p)
}


gwitdh = 800
gheight = 600


#small <- seq(10,500,10)
#large <- seq(1000,4000,1000)

sizeborder=1000

smoothmethod='loess'
smoothcolor='black'




#geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor)
#geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor)



#IndexSet
png(file = "index.png", bg = "transparent",gwitdh, gheight, units = "px")
is = read_csv("timeIndex.csv")
is_melted = melt(smooth(is,c('List','Set')),id='Size',variable.name = "Type")
p1 <- ggplot(is_melted,aes(x=Size,y=value,linetype = Type)) + theme(legend.position = "bottom") +ggtitle('Calculate complementary set') + ylab("Time in s") + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) #+  theme(axis.text.x= element_blank())
p2 <- ggplot(is_melted[is_melted$Type == 'Set',],aes(x=Size,y=value,linetype = Type))  + theme(legend.position = "bottom") + ylab("Time in s") + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor)
multiplot(p1, p2, cols=2)
dev.off()


#Subsetting-------------------------------------------------------------------------------------------------

#Dense-------------------------------------------------------------------------------------------------

  timeNPTake = read_csv("timeNPTake.csv")
  timeNPTake_melted <- melt(smooth(timeNPTake,cols=list('ix','take')),id='Size',variable.name = 'Type')
  p1 <- ggplot(timeNPTake_melted,aes(x=Size,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Subsetting Numpy ndarray') + ylab("Time in s")
  p2 <- ggplot(timeNPTake_melted[timeNPTake_melted$Type == 'take',],aes(x=Size,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s")
  

  timeNPTakeSparse = read_csv("atomicOperations.csv")[,c('Dim','Sparsity','sparseSubsetFI','sparseSubsetIX')]
  
  #png(file = "subsetDenseHeat.png", bg = "transparent",gwitdh, gheight, units = "px")
  p3 <- plot3d(timeNPTakeSparse,c('Dim','Sparsity'),'sparseSubsetFI','Fancy-Indexing on sparse matrix')
  p4 <- plot3d(timeNPTakeSparse,c('Dim','Sparsity'),'sparseSubsetIX',  'ix on sparse matrix')

  png(file = "subset.png", bg = "transparent",gwitdh, gheight, units = "px")
  multiplot(p1, p2,p3,p4, cols=2)
  dev.off()
  
  #Multiplication-------------------------------------------------------------------------------------------------
  ao = read_csv("atomicOperations.csv")
  mult_melted <- melt(smooth(ao,colnames(ao)[colnames(ao) != c('Dim','Sparsity')]),id='Dim',variable.name = 'Type')

  p1 <- ggplot(mult_melted[(mult_melted$Type == 'denseMatrixDotVector'& mult_melted$Dim<sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Calculate dot product with Dense Matrix') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  p3 <- ggplot(mult_melted[(mult_melted$Type == 'denseMatrixDotVector'& mult_melted$Dim>=sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Calculate dot product with Dense Matrix') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  

  p2 <- plot3d(ao[ao$Dim < sizeborder,],c('Dim','Sparsity'),'sparseMatrixDotVectorWithScipySparse','Dotproduct of sparse matrix and vector')
  p4 <- plot3d(ao[ao$Dim >= sizeborder,],c('Dim','Sparsity'),'sparseMatrixDotVectorWithScipySparse','Dotproduct of sparse matrix and vector')


  #p2 <- ggplot(mult_melted[(mult_melted$Type == 'sparseMatrixDotVectorWithScipySparse'&mult_melted$Dim<sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s") +ggtitle('Calculate dot product with Sparse Matrix')
  #p4 <- ggplot(mult_melted[(mult_melted$Type == 'sparseMatrixDotVectorWithScipySparse'&mult_melted$Dim>=sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s") +ggtitle('Calculate dot product with Sparse Matrix')
  
  
  png(file = "multVec.png", bg = "transparent",gwitdh, gheight, units = "px")
  multiplot(p1, p2,p3,p4, cols=2)
  dev.off()
  
  p1 <- ggplot(mult_melted[(mult_melted$Type == 'denseMatrixDotMatrix'& mult_melted$Dim<sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Calculate dot product with Dense Matrix') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  p3 <- ggplot(mult_melted[(mult_melted$Type == 'denseMatrixDotMatrix'& mult_melted$Dim>=sizeborder),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Calculate dot product with Dense Matrix') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  

  p2 <- plot3d(ao[ao$Dim < sizeborder,],c('Dim','Sparsity'),'sparseMatrixDotMatrix','Dotproduct of sparse matrices')
  p4 <- plot3d(ao[ao$Dim >= sizeborder,],c('Dim','Sparsity'),'sparseMatrixDotMatrix','Dotproduct of sparse matrices')
  
  png(file = "multMat.png", bg = "transparent",gwitdh, gheight, units = "px")
  multiplot(p1, p2,p3,p4, cols=2)
  dev.off()


  #Inversion-------------------------------------------------------------------------------------------------
  p1 <- ggplot(mult_melted[((mult_melted$Type == 'inversionTimeToDenseSplit' | mult_melted$Type == 'inversionTimeSparseCSC'| mult_melted$Type == 'inversionTimeSparseToDense') & mult_melted$Dim<100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Sparse Matrix Inversion : methods') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  p2 <- ggplot(mult_melted[((mult_melted$Type == 'inversionTimeToDenseSplit' | mult_melted$Type == 'inversionTimeSparseCSC'| mult_melted$Type == 'inversionTimeSparseToDense')&mult_melted$Dim>=100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s") +ggtitle('Sparse Matrix Inversion : methods')
  
  p3 <- ggplot(mult_melted[(mult_melted$Type == 'inversionTimeDense'& mult_melted$Dim<100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Dense Matrix Inversion') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  p4 <- ggplot(mult_melted[(mult_melted$Type == 'inversionTimeToDenseSplit'&mult_melted$Dim<100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s") +ggtitle('Sparse Matrix Inversion')
  p5 <- ggplot(mult_melted[(mult_melted$Type == 'inversionTimeDense'& mult_melted$Dim>=100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Dense Matrix Inversion') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
  p6 <- ggplot(mult_melted[(mult_melted$Type == 'inversionTimeToDenseSplit'&mult_melted$Dim>=100),],aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") + ylab("Time in s") +ggtitle('Sparse Matrix Inversion')
  
  
  png(file = "inv.png", bg = "transparent",gwitdh, gheight, units = "px")
  multiplot(p1,p3,p5,p2,p4,p6,cols=2)
  dev.off()
  

  p1 <- plot3d(ao,c('Dim','Sparsity'),'inversionTimeSparse','Inversion of matrix in sparse representation')
  p2 <- plot3d(ao,c('Dim','Sparsity'),'inversionTimeDense','Inversion of matrix in dense representation')

  png(file = "inv.png", bg = "transparent",gwitdh, gheight, units = "px")
  multiplot(p1,p2,cols=1)
  dev.off()


plotSurfacePlotly <- function(df,x,y,column){
  dims = unique(df[[axis[1]]])
  #print(length(dims))
  print(dims)
  df[[axis[2]]] <- round(df[[axis[2]]],digits = 1)
  sparsities <-unique(df[[axis[2]]])
  
  print(sparsities)
  
  s <- split(df,df['Dim'])
  
  map <-data.frame(
    lapply(1:(length(dims)),
           function(i){
             s[[i]][[column]]
           })
  )
  
  colnames(map) <- dims
  rownames(map) <- sparsities
}

plot3dplotly <- function(df,axis,column,maintitle,savename){
  dims = unique(df[[axis[1]]])
  #print(length(dims))
  print(dims)
  df[[axis[2]]] <- round(df[[axis[2]]],digits = 1)
  sparsities <-unique(df[[axis[2]]])
  
  print(sparsities)
  
  s <- split(df,df['Dim'])

  map <-data.frame(
    lapply(1:(length(dims)),
           function(i){
             s[[i]][[column]]
           })
  )
  
  map <- data.matrix(map)
  #colnames(map) <- dims
  #rownames(map) <- sparsities
  
  scene=list(camera=list(up=list(x=0,y=0,z=2), center=list(x=0,y=0,z=0),eye=list(x=2,y=2,z=0)))
  m <- list(
    l = 50,
    r = 50,
    b = 100,
    t = 100,
    pad = 4
  )
  
  axis <- list(
    autotick = FALSE,
    ticks = "outside",
    tick0 = 0,
    dtick = 0.25,
    ticklen = 5,
    tickwidth = 2,
    tickcolor = toRGB("blue")
    )
  
  axis <- list(
    autotick = FALSE,
    ticks = "outside",
    tick0 = 0,
    dtick = 0.25,
    ticklen = 5,
    tickwidth = 2,
    tickcolor = toRGB("blue")
    )
  
  p <- plot_ly(x =dims, y = sparsities, z = ~map)  %>% add_surface()  %>% layout(scene=scene,xaxis = axis, yaxis = axis,title=maintitle)
  
  plotly_IMAGE(p, format = "png", out_file = paste(savename,'.png'))
}


#Inference:
inferenceOperationsDense = read_csv("inferenceOperationsDense.csv")
inferenceOperationsDense_melted <- melt(smooth(inferenceOperationsDense,cols=list('mmcm','mmcc','mccm','mccc')),id='Dim',variable.name = 'Type')
inferenceOperationsSparse = read_csv("inferenceOperationsSparse.csv")
inferenceOperationsSparse_melted <- melt(smooth(inferenceOperationsSparse,cols=list('mmcm','mmcc','mccm','mccc')),id='Dim',variable.name = 'Type')

p1 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'mmcm','Marg: Mean,       Con: Mean,      Dense Representation')
p2 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'mmcc','Marg: Mean,       Con: Canonical, Dense Representation')
p3 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'mccm','Marg: Canonical,  Con: Mean,      Dense Representation')
p4 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'mccc','Marg: Canonical,  Con: Canonical, Dense Representation')
png(file = "denseOperationsSmall.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,p3,p4,cols=2)
dev.off()

p1 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'mmcm','Marg: Mean,       Con: Mean,      Dense Representation')
p2 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'mmcc','Marg: Mean,       Con: Canonical, Dense Representation')
p3 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'mccm','Marg: Canonical,  Con: Mean,      Dense Representation')
p4 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'mccc','Marg: Canonical,  Con: Canonical, Dense Representation')
png(file = "denseOperationsBig.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,p3,p4,cols=2)
dev.off()

#p1 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'mmcm','Marg: Mean,       Con: Mean,      Sparse Representation')
p2 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'mmcc','Marg: Mean,       Con: Canonical, Sparse Representation')
p3 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'mccm','Marg: Canonical,  Con: Mean,      Sparse Representation')
p4 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'mccc','Marg: Canonical,  Con: Canonical, Sparse Representation')
png(file = "sparseOperationsSmall.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p2,p3,p4,cols=1)
dev.off()

#p1 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'mmcm','Marg: Mean,       Con: Mean,      Sparse Representation')
p2 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'mmcc','Marg: Mean,       Con: Canonical, Sparse Representation')
p3 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'mccm','Marg: Canonical,  Con: Mean,      Sparse Representation')
p4 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'mccc','Marg: Canonical,  Con: Canonical, Sparse Representation')
png(file = "sparseOperationsBig.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,p3,cols=1)
dev.off()


p1 <- ggplot(inferenceOperationsDense_melted[(
      inferenceOperationsDense_melted$Type == 'mmcm' | 
      inferenceOperationsDense_melted$Type == 'mmcc'| 
      inferenceOperationsDense_melted$Type == 'mccm'| 
      inferenceOperationsDense_melted$Type == 'mccc') & 
      inferenceOperationsDense_melted$Dim<sizeborder,],
      aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Inference on dense representation') + ylab("Time in s") #+  theme(axis.text.x= element_blank())

p2 <- ggplot(inferenceOperationsDense_melted[(
      inferenceOperationsDense_melted$Type == 'mmcm' | 
      inferenceOperationsDense_melted$Type == 'mmcc') & 
      inferenceOperationsDense_melted$Dim<sizeborder,],
      aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Inference on dense representation') + ylab("Time in s") #+  theme(axis.text.x= element_blank())

png(file = "denseOperationsLine.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,cols=2)
dev.off()

p1 <- ggplot(inferenceOperationsSparse_melted[(
  inferenceOperationsSparse_melted$Type == 'mmcm' | 
    inferenceOperationsSparse_melted$Type == 'mmcc'| 
    inferenceOperationsSparse_melted$Type == 'mccm'| 
    inferenceOperationsSparse_melted$Type == 'mccc') & 
    inferenceOperationsSparse_melted$Dim<sizeborder,],
  aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Inference on mixed representation') + ylab("Time in s") #+  theme(axis.text.x= element_blank())

p2 <- ggplot(inferenceOperationsSparse_melted[(
  inferenceOperationsSparse_melted$Type == 'mmcm' | 
    inferenceOperationsSparse_melted$Type == 'mmcc') & 
    inferenceOperationsSparse_melted$Dim<sizeborder,],
  aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Inference on mixed representation') + ylab("Time in s") #+  theme(axis.text.x= element_blank())

png(file = "sparseOperationsLine.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,cols=2)
dev.off()

#ConditionOnly
inferenceOperationsDense = read_csv("inferenceOperationsDense.csv")
inferenceOperationsDense_melted <- melt(inferenceOperationsDense,id='Dim',variable.name = 'Type')
inferenceOperationsSparse = read_csv("inferenceOperationsSparse.csv")
inferenceOperationsSparse_melted <- melt(inferenceOperationsSparse,id='Dim',variable.name = 'Type')


p1 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, dense representation')
p2 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim < sizeborder,],c('Dim','Sparsity'),'conditionOnlyMean','Conditioning on mean form, dense representation')
p3 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, dense representation')
p4 <- plot3d(inferenceOperationsDense[inferenceOperationsDense$Dim >= sizeborder,],c('Dim','Sparsity'),'conditionOnlyMean','Conditioning on mean form, dense representation')
p5 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, mixed representation')
p6 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, mixed representation')


png(file = "conditionAll.png", bg = "transparent",gwitdh, gheight, units = "px")
multiplot(p1,p2,p3,p5,p4,p6, cols=2)
dev.off()


#p1 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, mixed representation')
#p2 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim < sizeborder,],c('Dim','Sparsity'),'conditionOnlyMean','Conditioning on mean form, mixed representation')
#p3 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'conditionOnlyCanonical','Conditioning on canonical form, mixed representation')
#p4 <- plot3d(inferenceOperationsSparse[inferenceOperationsSparse$Dim >= sizeborder,],c('Dim','Sparsity'),'conditionOnlyMean','Conditioning on mean form, mixed representation')
#png(file = "conditionSparse.png", bg = "transparent",gwitdh, gheight, units = "px")
#multiplot(p1,p3,cols=1)
#dev.off()

#Best of Condtition
p1 <- ggplot(inferenceOperationsSparse_melted[(
  inferenceOperationsSparse_melted$Type == 'mmcm' | 
    inferenceOperationsSparse_melted$Type == 'mmcc'| 
    inferenceOperationsSparse_melted$Type == 'mccm'| 
    inferenceOperationsSparse_melted$Type == 'mccc') & 
    inferenceOperationsSparse_melted$Dim<sizeborder,],
  aes(x=Dim,y=value,linetype = Type)) + geom_smooth(method=smoothmethod,formula=y~x,color = smoothcolor) + theme(legend.position = "bottom") +ggtitle('Inference on mixed representation') + ylab("Time in s") #+  theme(axis.text.x= element_blank())
multiplot(p1,p2,cols=2)















