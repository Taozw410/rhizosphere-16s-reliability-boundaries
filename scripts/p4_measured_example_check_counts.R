counts <- as.matrix(read.csv('data/formal_measure/input/counts.csv', row.names=1, check.names=FALSE))
storage.mode(counts) <- 'integer'
cat('dims',dim(counts),'anyNA',anyNA(counts),'range',range(counts),'row zero',sum(rowSums(counts)==0),'col zero',sum(colSums(counts)==0),'\n')
cat('class',class(counts),'mode',storage.mode(counts),'\n')
