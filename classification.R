install.packages("data.table")
install.packages("mlr3verse")
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("ggforce")
install.packages("randomForest")
install.packages("caret")
install.packages("DataExplorer")

library("data.table")
library("mlr3verse")
library("tidyverse")
library("ggplot2")
library("ggforce")
require("randomForest")
library("caret")


loans <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
View(loans)

skimr::skim(loans)


dim(loans)

table(loans$Personal.Loan)

DataExplorer::plot_bar(loans, ncol = 3)

DataExplorer::plot_histogram(loans, ncol = 3)

DataExplorer::plot_boxplot(loans, by = "Personal.Loan", ncol = 3)

# It was observed that there were some experience that were negative -> not reasonable
cor(loans)

sum(loans$Experience<0)


loans.p <- dplyr::select(loans, Experience, Age, Income, CCAvg, Mortgage)
pairs(loans.p)

#drop experience as there were negative value in Experience

loans <- select(loans, -Experience) %>%
          mutate(Investment.account = case_when(
          Securities.Account + CD.Account > 0 ~ "1",
          TRUE ~ "0")) %>%
          select(-Securities.Account, -CD.Account)
          
loans.par <-loans %>%
            select(Personal.Loan, Online, CreditCard, Investment.account) %>%
            mutate(Online = case_when(
                   Online >0 ~ "Online", 
                  TRUE ~ "None" ))%>%
            mutate(CreditCard = case_when(
                   CreditCard >0 ~ "CreditCard", 
                  TRUE ~ "None" ))%>%
            mutate(Investment.account = case_when(
                   Investment.account >0 ~ "Investment", 
                  TRUE ~ "None" ))%>%
            group_by(Personal.Loan, Online, CreditCard, Investment.account) %>%
            summarize(value = n())
            
library("ggforce")



ggplot(loans.par %>% gather_set_data(x = c(2:4)),
       aes(x = x, id = id, split = y, value = value)) +
  geom_parallel_sets(aes(fill = as.factor(Personal.Loan)),
                     axis.width = 0.1,
                     alpha = 0.66) + 
  geom_parallel_sets_axes(axis.width = 0.15, fill = "lightgrey") + 
  geom_parallel_sets_labels(angle = 0) +
  coord_flip()
  
  
loans$Personal.Loan = as.factor(loans$Personal.Loan)
loans$Investment.account = as.numeric(loans$Investment.account)


set.seed(100) # set seed for reproducibility
loans_task <- TaskClassif$new(id = "Bankloans",
                               backend = loans,
                               target = "Personal.Loan",
                               positive = '1')
                               
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(loans_task)



lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(loans_task),
  learner    = list(lrn_baseline,
                    lrn_cart),
  resampling = list(cv5)
), store_models = TRUE)

autoplot(res, type = "roc")

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))




plot(res$resample_result(2)$learners[[1]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[1]]$model, use.n = TRUE, cex = 0.6)
plot(res$resample_result(2)$learners[[2]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[2]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[3]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[3]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[4]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[4]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)


# Pruning tree
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)

res_cart_cv <- resample(loans_task, lrn_cart_cv, cv5, store_models = TRUE)


rpart::printcp(res_cart_cv$learners[[1]]$model)
rpart::plotcp(res_cart_cv$learners[[1]]$model)
rpart::printcp(res_cart_cv$learners[[2]]$model)
rpart::plotcp(res_cart_cv$learners[[2]]$model)
rpart::printcp(res_cart_cv$learners[[3]]$model)
rpart::plotcp(res_cart_cv$learners[[3]]$model)
rpart::printcp(res_cart_cv$learners[[4]]$model)
rpart::plotcp(res_cart_cv$learners[[4]]$model)
rpart::printcp(res_cart_cv$learners[[5]]$model)
rpart::plotcp(res_cart_cv$learners[[5]]$model)


lrn_cart_cp <- lrn("classif.rpart", predict_type = "prob", cp = 0.044)
# lrn_rf <- lrn("classif.ranger", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(loans_task),
  learner    = list(lrn_baseline,
                    lrn_cart, 
                    lrn_cart_cp),
  resampling = list(cv5)
), store_models = TRUE)

# autoplot(res, type = "roc")
# autoplot(res, type = "prc")

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
                   
                   
plot(res$resample_result(2)$learners[[1]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[1]]$model, use.n = TRUE, cex = 0.6)
plot(res$resample_result(2)$learners[[2]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[2]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[3]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[3]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[4]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[4]]$model, use.n = TRUE, cex = 0.8)
plot(res$resample_result(2)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(2)$learners[[5]]$model, use.n = TRUE, cex = 0.8)


bootstrap <- rsmp("bootstrap")
bootstrap$instantiate(loans_task)


res <- benchmark(data.table(
  task       = list(loans_task),
  learner    = list(lrn_baseline,
                    lrn_cart, 
                    lrn_cart_cp),
  resampling = list(bootstrap)
), store_models = TRUE)



res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
                   
                   
lrn_rf <- lrn("classif.ranger", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(loans_task),
  learner    = list(lrn_baseline,
                    lrn_cart_cp,
                    lrn_rf 
                    ),
  resampling = list(cv5)
), store_models = TRUE)

autoplot(res, type = "roc")
autoplot(res, type = "prc")

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


# Create a pipeline which encodes and then fits an XGBoost model
lrn_xgboost <- lrn("classif.xgboost", predict_type = "prob")
pl_xgb <- po("encode") %>>%
  po(lrn_xgboost)

# logistic Regression
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")


res <- benchmark(data.table(
  task       = list(loans_task),
  learner    = list(lrn_baseline,
                    lrn_log_reg,
                    lrn_cart_cp,
                    lrn_rf,
                    pl_xgb),
  resampling = list(cv5)
), store_models = TRUE)

autoplot(res, type = "roc")

autoplot(res, type = "prc")




res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))


require("randomForest")
fit=randomForest(factor(Personal.Loan)~., data=loans)

(VI_F=importance(fit))
library("caret")
varImp(fit)
varImpPlot(fit,type=2, main ="Random Forest")

  
