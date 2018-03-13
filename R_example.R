library(swat)
library(random)
library(wordcloud)
library(tm)

#connect to cas
sess <- swat::CAS('eeclxvm128b.unx.sas.com', 5570)
#load actionsets
loadActionSet(sess,'dataStep')
loadActionSet(sess,"dataPreprocess")
loadActionSet(sess,"cardinality")
loadActionSet(sess,"sampling")
loadActionSet(sess,"regression")
loadActionSet(sess,"decisionTree")
loadActionSet(sess,"neuralNet")
loadActionSet(sess,"svm")
loadActionSet(sess,"astore")
loadActionSet(sess,"percentile")
loadActionSet(sess,"clustering")
loadActionSet(sess,"textMining")

#load table
if(cas.table.tableExists(sess,table="final_debates_16")==FALSE) {
  tab1=cas.read.sas7bdat(sess,"/home/sasdemo/data/final_debates_16.sas7bdat")
}
if(cas.table.tableExists(sess,table="analysis_2016")==FALSE) {
  tab2=cas.read.sas7bdat(sess,"/home/sasdemo/data/analysis_2016.sas7bdat")
}
if(cas.table.tableExists(sess,table="engstop")==FALSE) {
  tab3=cas.read.sas7bdat(sess,"/home/sasdemo/data/engstop.sas7bdat")
}

#fetch first 5 rows
head(tab2)
cas.table.fetch(tab2,to=5)
cas.simple.summary(tab2)


#TextMining
res=cas.textMining.tmMine(sess,
                          documents=list(name='analysis_2016'),
                          stoplist="engstop",
                          docId="ID",
                          text="Text",
                          target="Speaker",
                          numTopics=25,
                          tagging=TRUE,
                          STEMMING=TRUE,
                          reduce=5,
                          entities="STD",
                          TermWeight="MI",
                          CellWeight="LOG",
                          selectEntity=list(opType="IGNORE",taglist="PERSON"),
                          selectPOS=list(opType="KEEP",taglist=list("NOUN","VERB","PROP","ADJ")),
                          terms=list(name="terms",replace=TRUE),
                          parent=list(name="parent",replace=TRUE),
                          topics=list(name="topics",replace=TRUE),
                          docPro=list(name="docPro",replace=TRUE),
                          copyVars=list("Speaker"))



cas.table.fetch(sess,table="terms",to=25)
cas.table.fetch(sess,table="topics",to=25)

topics <- defCasTable(sess, 'topics')
docPro <- defCasTable(sess, 'docPro')
terms <- defCasTable(sess, 'terms')


#head(topics,25)




####################ANALYZE THE TERMS
df=to.data.frame(to.casDataFrame(terms))

#Filtering by Frequency

df_filter=subset(df,`_Frequency_`>25 & `_Keep_`=="Y" & (`_IsPar_`=="+" | `_IsPar_`==""), select=c("_Role_","_Term_","_Frequency_","_Weight_"))
sorted <- df_filter[order(-df_filter$`_Frequency_`),] 
wordcloud(df_filter$`_Term_`,df$`_Frequency_`)

#ByWeight
wordcloud(df_filter$`_Term_`,df$`_Weight_`)

#ByWeight and Frequency
df_filter2=cbind(df_filter,Weight_Freq=df_filter$`_Frequency_`*df_filter$`_Weight_`)
wordcloud(df_filter2$`_Term_`,df_filter2$Weight_Freq)


#Persons
Persons=subset(df,`_Role_`=='PERSON', select=c("_Role_","_Term_","_Frequency_"))
wordcloud(Persons$`_Term_`,Persons$`_Frequency_`)

#Locations
Locations=subset(df,`_Role_`=='LOCATION' & `_Term_`!='well', select=c("_Role_","_Term_","_Frequency_"))
wordcloud(Locations$`_Term_`,Locations$`_Frequency_`)


#######Topics
head(topics,25)





#Modeling
head(docPro,5)
cas.sampling.stratified(sess,
                        table="docPro",
                        samppct=30,
                        PartInd=TRUE,
                        seed=12345,
                        Output=list(casOut=list(name="PartInd",replace=TRUE),copyVars="All",partindname="PartInd"))
parttab=defCasTable(sess, 'PartInd')


train=subset.casTable(parttab,PartInd==0,tname="train")
val=subset.casTable(parttab,PartInd==1,tname="val")


inputs=list("_COL1_","_COL2_","_COL3_","_COL4_","_COL5_","_COL6_","_COL7_","_COL8_","_COL9_","_COL10_",
            "_COL11_","_COL12_","_COL13_","_COL14_","_COL15_","_COL16_","_COL17_","_COL18_","_COL19_","_COL20_",
            "_COL21_","_COL22_","_COL23_","_COL24_","_COL25_")



#Forest
forest=cas.decisionTree.forestTrain(train,
                                    target   = "speaker",
                                    inputs   = inputs,
                                    leafSize=10,
                                    oob=TRUE,
                                    ntree=100,
                                    maxBranch=5, 
                                    maxLevel=10,
                                    varImp   = TRUE,
                                    casOut   = list(name = 'rf_model', replace = TRUE))
forest$ModelInfo

#get important variables
vi=forest$DTreeVarImpInfo
dftop=to.data.frame(to.casDataFrame(topics))
top=cbind(as.vector(dftop$`_Name_`),as.vector(inputs))
for(i in dim(top)[1]:1) {
char=paste ("_Col",i, sep = "", collapse = NULL)
vi[grepl(char,vi[,1]),1]=top[i,1]
}
varimp=cbind(vi)
varimp[,1:2]


#SVM
cas.table.dropTable(sess,table="svm_model")
svm=cas.svm.svmTrain(parttab,
                     partbyvar=list(name="PartInd", train="0", validate="1"),
                     target   = "speaker",
                     inputs   = inputs,
                     output   = list(casOut="svm_model",copyvars="ALL"))


svmout=defCasTable(sess,table="svm_model")
svm$Misclassification
svm$FitStatistics

#NEURAL
neural=cas.neuralNet.annTrain(train,
                              validTable=list(name="PartInd",where="PartInd=1"),
                              target   = "speaker",
                              inputs   = inputs,
                              targetStd="STD",
                              hiddens=list(5, 5, 5),
                              casOut   = list(name = 'nn_model', replace = TRUE))

cas.neuralNet.annScore(sess,
                       table=list(name="PartInd",where="PartInd=1"),
                       modelTable='nn_model',
                       casOut   = list(name = 'nn_valid', replace = TRUE))

