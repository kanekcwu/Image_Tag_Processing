
#------------------------------------------#
# Hack the Rack 2018                       #
# Challenge 2: Image Tag Processing        #
# Created by cyda - Yeung Wong & Carrie Lo #
#------------------------------------------#

# Please acknowledge team cyda - Yeung Wong and Carrie Lo when using the code
# If you find this script is helpful, please feel free to endorse us through Linkedin!

##########################################################
#                        Linkedin                        #
##########################################################
#                                                        #
# Yeung Wong - https://www.linkedin.com/in/yeungwong/    #
# Carrie Lo - https://www.linkedin.com/in/carrielsc/     #
#                                                        #
##########################################################

#----------------------------------------------------#
# Challenge: HUMANIZING IMAGE SEARCH FOR INSPIRATION # 
#----------------------------------------------------#

# This project is intentionally made to process the text data of the tagging of over 25,000 images
# provided by Li and Fung so as to make a search engine of their products more easily and effectively.
# Therefore, in order to facilitate their daily works, we drill down our challenge into two main focuses

# - Part 1: Clean the input dataset from two different APIs and create a Image_Tag master dataset
# - Part 2: Leverage pretrained neural network to enhance the customer search experience

# Example of the Image_Tag master dataset
#     | pic_id | tags     |
#     ---------------------
#     | pic001 | "dress"  |
#     | pic001 | "pink"   |
#     | pic001 | "summer" |
#     | pic002 | "hat"    |
#     | ...    |  ...     |

############
# Reminder #
############
#
# Please make sure you check the below checkpoints before running the script.
# 
# 1. This script is Part 1 of the challenge. For details in Part 2 which is worked in the R environment, you may check thought our github - cydalytics.
# 2. Make sure the file path is correct and the files are following the hirarchy. (Please refer 1.2 define the path for detail)

#-------------#
# Data we use #
#-------------#

# Input: 
# > Raw data - tagged_results.csv
# > Processing data - special_stopwords.csv

# Output:
# > Processed data - json_atr_tag_dataset.csv

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#---------------#
# 1 Preliminary #
#---------------#

# 1.1 import libraries and set global parameter

require(xlsx)
require(textreg)
require(tm)
require(reshape2)
require(tokenizers)
require(ngram)
require(openNLP)
require(CTM)
require(Rwordseg)
library(stringr)
library(corpus)

# 1.2 define the path

project = getwd()
raw_data_dir = paste(project,"/Data/Raw Data", sep="")
processing_data_dir = paste(project,"/Data/Processing Data", sep="")
processed_data_dir = paste(project,"/Data/Result & Processed Data", sep="")

#--------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------#
# 2 Text Cleansing #
#------------------#

# 2.1 data exploration

# There are two kinds of dataset reveived from the external companies - json and csv
# It is fine to extract features and do further analysis on the csv based data
# However, the json data are displayed as a string inside a cell which is hard to deal with
# In order to facilitate the demostration, we will focus on how we crack on the json-wise dataset

setwd(raw_data_dir)
raw_dataset = read.csv("tagged_results.csv")
json_label_dataset = raw_dataset[nchar(as.character(raw_dataset$attributes)) != 0,]
csv_label_dataset = raw_dataset[nchar(as.character(raw_dataset$attributes)) == 0,]

json_atr_dataset = data.frame(cbind(as.character(json_label_dataset$filename), as.character(json_label_dataset$attributes)))
colnames(json_atr_dataset) = c("filename", "attributes")
json_atr_dataset = json_atr_dataset[nchar(as.character(json_atr_dataset$attributes)) != 2,] # remove those untag cases

# 2.2 text cleansing

# Actually, we have explored and tried different text cleansing technique such as tokenize the phrases into unigram / bigram
# But it comes to a critical problem which is "multiple negation" problem
# In some sense it just means there are many irrelevant or useless taggings will be generated
# We finally come up with a simple but strongly effective idea on the text cleansing part
#--------------------------------------------------------------------------------------------------#
# create a stopword list beforehead -> preprocessing by formatting all words into small letters -> #
# remove the stopwords -> observe pattern and output the tag                                       #
#--------------------------------------------------------------------------------------------------#
# Remark: we did not do further text cleansing on the output taggings based on two reasons         #
#         - we want to keep the format that different companies use                                #
#         - we have a synonym model that cater the problem                                         #
#--------------------------------------------------------------------------------------------------#
# If you are interested what technique we have discovered, please check in the appendix section

# import stopword list
setwd(processing_data_dir)
special_stopwords = read.csv("special_stopwords.csv")
special_stopwords_mod = special_stopwords[order(-nchar(as.character(special_stopwords$Remove))),] # to tackle the 'multiple negation' removing problem

# format into small letter
json_atr_dataset$attributes = tolower(json_atr_dataset$attributes)

# remove stopwords
for(i in 1:nrow(special_stopwords_mod))
{
  json_atr_dataset$attributes <- gsub(special_stopwords_mod[i,1],special_stopwords_mod[i,2], json_atr_dataset$attributes)
}

# observe pattern and output the tag
json_atr_tag_dataset = NULL
for (i in c(1:nrow(json_atr_dataset)))
{
  for (j in c(1:(length(gregexpr(": ",json_atr_dataset$attributes[i])[[1]])-1)))
  {
    temp_atr_value = substr(json_atr_dataset$attributes[i], gregexpr(": ",json_atr_dataset$attributes[i])[[1]][j]+4, gregexpr(": ",json_atr_dataset$attributes[i])[[1]][j+1]-8)
    json_atr_tag_dataset = rbind(json_atr_tag_dataset, data.frame(`filename`= json_atr_dataset$filename[i], `atr_tag` = temp_atr_value))
  }
  temp_atr_value = substr(json_atr_dataset$attributes[i], gregexpr(": ",json_atr_dataset$attributes[i])[[1]][length(gregexpr(": ",json_atr_dataset$attributes[i])[[1]])]+4, nchar(json_atr_dataset$attributes[i])-3)
  json_atr_tag_dataset = rbind(json_atr_tag_dataset, data.frame(`filename`= json_atr_dataset$filename[i], `atr_tag` = temp_atr_value))
}
json_atr_tag_dataset <- json_atr_tag_dataset[!(json_atr_tag_dataset$atr_tag == ""),] # remove those empty attribute cases

# save the dataset

# json_atr_tag_dataset is now having the nice format that we want as mentioned in the challenge discription part
# To save the running time, in the following analysis, we will use this dataset as the master dataset
# For the company purpose, it is highly suggested to include those tags in "category", "sub_category" and "product_type"
# It is simply and easy as all of them are in structured table form already =)

setwd(processed_data_dir)
write.csv(json_atr_tag_dataset, "json_atr_tag_dataset.csv")
#--------------------------------------------------------------------------------------------------------------------------------------------------------

#------------#
# 3 Appendix #
#------------#

# This is the part showing the coding that we have used during the data exploration and manipulation time
# We think it is a pity if just delete them since they are useful techniques
# And it is hard to explain one by one why we do not include them into the script
# So, if you want to know more about it, feel free to contact us through Linkedin and have a fun discussion

#-------------------------------------------------#
# Explore the json dataset with different columns # (For company to build the full solution for json dataset)
#-------------------------------------------------#
# json_cat_dataset = data.frame(cbind(as.character(json_label_dataset$filename), as.character(json_label_dataset$category)))
# colnames(json_cat_dataset) = c("filename", "category")
# json_cat_dataset = json_cat_dataset[!(json_cat_dataset$category == ""),]
# json_product_type_dataset = data.frame(cbind(as.character(json_label_dataset$filename), as.character(json_label_dataset$product_type)))
# colnames(json_product_type_dataset) = c("filename", "product_type")
# json_product_type_dataset = json_product_type_dataset[!(json_product_type_dataset$product_type == ""),]
# json_sub_cat_dataset = data.frame(cbind(as.character(json_label_dataset$filename), as.character(json_label_dataset$sub_category)))
# colnames(json_sub_cat_dataset) = c("filename", "sub_category")
# json_sub_cat_dataset = json_sub_cat_dataset[!(json_sub_cat_dataset$sub_category == ""),]

#--------#
# n_gram #
#--------#
# n_gram <- function(x, n){
#   unlist(lapply(ngrams(words(x), n), paste, collapse = " "), use.names = FALSE)
# }

#------------#
# multi_gram #
#------------#
# multigram <- function(x){
#   unlist(list(
#     unlist(lapply(ngrams(words(x), 1), paste, collapse = " "), use.names = FALSE),
#     unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE),
#     unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
#   ))
# }

#-------#
# TFIDF #
#-------#
# tfidf <- function(docs){
#   # Document Term Matrix
#   dtm_uni <- DocumentTermMatrix(docs
#                                 , control = list(weighting = weightTfIdf)
#   )
#   dtm_bi <- DocumentTermMatrix(docs
#                                , control = list(tokenize = function(x) n_gram(x, 2)
#                                                 , weighting = weightTfIdf
#                                ))
#   dtm_tri <- DocumentTermMatrix(docs
#                                 , control = list(tokenize = function(x) n_gram(x, 3)
#                                                  , weighting = weightTfIdf
#                                 ))
#   # Word Frequency
#   freq1 <- colSums(as.matrix(dtm_uni))
#   freq2 <- colSums(as.matrix(dtm_bi))
#   freq3 <- colSums(as.matrix(dtm_tri))
#   ft1 = data.frame(frequency = freq1[order(freq1, decreasing = T)])
#   ft2 = data.frame(frequency = freq2[order(freq2, decreasing = T)])
#   ft3 = data.frame(frequency = freq3[order(freq3, decreasing = T)])
#   
#   out <- list(unigram = ft1, bigram = ft2, trigram = ft3, 
#               dtm_uni = dtm_uni, dtm_bi = dtm_bi, dtm_tri = dtm_tri)
#   
#   return(out)  
# }

#----------------#
# term frequency #
#----------------#
# tf <- function(docs){
#   # Document Term Matrix
#   dtm_uni <- DocumentTermMatrix(docs)
#   dtm_bi <- DocumentTermMatrix(docs
#                                , control = list(tokenize = function(x) n_gram(x, 2)))
#   dtm_tri <- DocumentTermMatrix(docs
#                                 , control = list(tokenize = function(x) n_gram(x, 3)))
#   # Word Frequency
#   freq1 <- colSums(as.matrix(dtm_uni))
#   freq2 <- colSums(as.matrix(dtm_bi))
#   freq3 <- colSums(as.matrix(dtm_tri))
#   ft1 = data.frame(frequency = freq1[order(freq1, decreasing = T)])
#   ft2 = data.frame(frequency = freq2[order(freq2, decreasing = T)])
#   ft3 = data.frame(frequency = freq3[order(freq3, decreasing = T)])
#   
#   out <- list(unigram = ft1, bigram = ft2, trigram = ft3, 
#               dtm_uni = dtm_uni, dtm_bi = dtm_bi, dtm_tri = dtm_tri)
#   
#   return(out)
# }

#---------------------------#
# term frequency percentage #
#---------------------------#
# tfp <- function(docs){
#   # Document Term Matrix
#   dtm_uni <- DocumentTermMatrix(docs)
#   dtm_bi <- DocumentTermMatrix(docs
#                                , control = list(tokenize = function(x) n_gram(x, 2)))
#   dtm_tri <- DocumentTermMatrix(docs
#                                 , control = list(tokenize = function(x) n_gram(x, 3)))
#   # Word Frequency
#   freq1 <- colSums(as.matrix(dtm_uni))
#   freq2 <- colSums(as.matrix(dtm_bi))
#   freq3 <- colSums(as.matrix(dtm_tri))
#   ft1 = data.frame(frequency = freq1, percentage = freq1/length(docs))
#   ft1 <- ft1[order(freq1, decreasing = T),][1:50,]
#   ft2 = data.frame(frequency = freq2, percentage = freq2/length(docs))
#   ft2 <- ft2[order(freq2, decreasing = T),][1:50,]
#   ft3 = data.frame(frequency = freq3, perentage = freq3/length(docs))
#   ft3 <- ft3[order(freq3, decreasing = T),][1:50,]
#   
#   out <- list(length = length(docs), unigram = ft1, bigram = ft2, trigram = ft3)
#   
#   return(out)
# }

#----------------#
# text cleansing #
#----------------#
# cleansing_text <- function(text){
#   doc <- Corpus(VectorSource(text))
#   doc <- tm_map(doc, PlainTextDocument)
#   doc <- tm_map(doc, stripWhitespace)
#   doc <- tm_map(doc, removePunctuation)
#   doc <- tm_map(doc, removeNumbers)
#   
#   # Handling Emoji Problem
#   # Ignore the error is okay
#   suppressWarnings(
#     for (i in c(1:length(doc$content$content)))
#     {
#       doc$content$content[i] = str_replace_all(doc$content$content[i],"[^[:graph:]]", " ") 
#       t <- try(tolower(doc$content$content[i]))
#       if(("try-error"%in% class(t))) 
#       {
#         doc$content$content[i] <- NA
#       }
#     })
#   
#   doc <- tm_map(doc, tolower)
#   
#   ### modify stopwords - exclude but, no, not
#   stopwords1 <- stopwords("en")[stopwords("en") != "but"]
#   stopwords1 <- stopwords1[stopwords1 != "no"]
#   stopwords1 <- stopwords1[stopwords1 != "not"]
#   doc <- tm_map(doc, removeWords, stopwords1)
#   doc <- tm_map(doc, stripWhitespace)
#   doc <- tm_map(doc, PlainTextDocument)
#   
#   # Stemming english words
#   for(i in 1:nrow(attributes)){
#     doc$content$content <- gsub(attributes[i,1], 
#                                 attributes[i,2], doc$content$content)
#   }
#   doc <- tm_map(doc, stripWhitespace)
#   doc <- tm_map(doc, PlainTextDocument)
#   
#   new_txt <- doc$content$content #extract character texts from Corpus
#   new_txt <- gsub('\uFEFF', '', new_txt)
#   new_txt <- trim(new_txt)
#   new_txt <- ifelse(new_txt == ' ', '', new_txt)
#   new_txt <- ifelse(nchar(new_txt) >= 1, new_txt, '')
#   
#   return(new_txt)
# }
# 
# trim <- function (x) gsub("^\\s+|\\s+$", "", x)

#-----------------------------------#
# ~ This is the end of the script ~ #
#-----------------------------------#

#--------------------------------------------------------------------------------------------------------------------------------------------------------

# If you appreciate our hard work, please endorse us through linkedin!

##########################################################
#                        Linkedin                        #
##########################################################
#                                                        #
# Yeung Wong - https://www.linkedin.com/in/yeungwong/    #
# Carrie Lo - https://www.linkedin.com/in/carrielsc/     #
#                                                        #
##########################################################