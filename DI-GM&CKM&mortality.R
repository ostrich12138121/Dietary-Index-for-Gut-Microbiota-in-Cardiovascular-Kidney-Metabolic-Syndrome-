library(nhanesR)
library(survey)
library(survival)
tsv <- nhs_tsv('demo',years = 2007:2017)
x1<- nhs_read(tsv,
              'wtint2yr',
              'wtmec2yr',
              'ridageyr:age',
              'riagendr:sex',
              'ridreth1:race',
              'dmdmartl:marr',
              'indfmpir:poor', 
              codebook=TRUE,psu_strat = T)
x1$race <- Recode(x1$race,
                  'Other Race - Including Multi-Racial::other',
                  'Other Hispanic::other',
                  to.numeric=FALSE)
x1 <- add_col(data = x1,colname = 'PIR',value = '≤1.00',condition = x1$poor<=1)
x1 <- add_col(data = x1,colname = 'PIR',value = '1.0-3.0',condition = x1$poor>1&x1$poor<=3)
x1 <- add_col(data = x1,colname = 'PIR',value = '＞3.0',condition = x1$poor>3)
x1$marr <- Recode(x1$marr,
                  'Living with partner::Married',
                  'Separated::Divorced',
                  'Widowed::Divorced',
                  to.numeric=FALSE)
x2<- db_demo(years = 2007:2017,edu=TRUE,psu_strat = F)
x2$edu <- Recode(x2$edu,
                 "High School Grad/GED or Equivalent::High school or equivalent",
                 "Some College or AA degree::College or above",
                 '9-11th Grade (Includes 12th grade with no diploma)::High school or equivalent',
                 'Less Than 9th Grade::Less than high school',
                 "College Graduate or above::College or above",
                 "High school graduate/GED or equivalent::High school or equivalent",
                 'Some college or AA degree::College or above',
                 'College graduate or above::College or above',
                 '9-11th grade (Includes 12th grade with no diploma)::High school or equivalent',
                 'Less than 9th grade::Less than high school',
                 to.numeric=FALSE)
z<- nhs_tsv('bmx',years = 2007:2017)
x3<- nhs_read(z,'bmxbmi:bmi',psu_strat = F) 
x3 <- add_col(data = x3,colname = 'BMI',value = 'Normal',condition = x3$bmi<25)
x3 <- add_col(data = x3,colname = 'BMI',value = 'Overweight',condition = x3$bmi>=25&x3$bmi<30)
x3 <- add_col(data = x3,colname = 'BMI',value = 'Obese',condition = x3$bmi>=30)
x4<- diag_smoke(
    years = 2007:2017,
    smoke = TRUE,
    never = 0,
    former = 1,
    now = 2)
x5<- diag_alcohol.user(
    years = 2007:2017,
    mild = c(1, 2),
    moderate = c(2, 3),
    heavy = c(3, 4),
    binge = TRUE)
tht<- nhs_tsv('bpq',years = 2007:2017)
x6<- nhs_read(tht,'bpq020:toldht',psu_strat = F)
x7<- diag_Hyperlipidemia(years = 2007:2017, yes1 = FALSE)
x8<- diag_DM(
    years = 2007:2017,
    told = TRUE,
    HbA1c = TRUE,
    fast_glu = TRUE,
    OGTT2 = TRUE,
    rand_glu = TRUE,
    drug = TRUE,
    DM1 = FALSE,
    exclude_Pregnant = TRUE)
x9<- diag_CKD(
    years = 2007:2017,
    show_CKD = TRUE,
    eGFR_method = "CKD_EPI_Scr_2009")
x10<- dex_PhysicalActivity(
    years = 2007:2017,
    Muscle.strength = T,
    RecreationalActivity = T,
    activity = T)
x10$recreational.activity <- Recode(x10$recreational.activity,
                                    "moderate::yes",
                                    "both::yes",
                                    "vigorous::yes")
x11<- dex_DI_GM(
    years = 2007:2017,
    day = c(1,2),
    score = TRUE,
    component = F)
x12<- mort_read(years = 2007:2017, varLabel = FALSE, codebook = TRUE)
col_rename(x12) <- c('mortstat:status','ucod_leading:cause','permth_exm:time')
x12$status <- Recode(x12$status,'Assumed deceased::1','Assumed alive::0',to.numeric = T)
x12$cause <- Recode(x12$cause,
                    "Accidents (unintentional injuries) (V01-X59, Y85-Y86)::Accidents" ,
                    "All other causes (residual)::other" ,
                    "Alzheimer's disease (G30)::AD" ,
                    "Cerebrovascular diseases (I60-I69)::brain" ,
                    "Chronic lower respiratory diseases (J40-J47)::lung" ,
                    "Diabetes mellitus (E10-E14)::DM" ,
                    "Diseases of heart (I00-I09, I11, I13, I20-I51)::heart" ,
                    "Influenza and pneumonia (J09-J18)::Ip",
                    "Malignant neoplasms (C00-C97)::Cancer",
                    "Nephritis, nephrotic syndrome and nephrosis (N00-N07, N17-N19, N25-N27)::kidney" ,
                    "NA::no",
                    to.numeric=FALSE)
j<- nhs_tsv('csx_h',years = 2013)
x13<- nhs_read(j,'csxquisg:quinine','csxquist:nobitte',psu_strat = F) 
x14<- db_drtot(years = 2007:2017,
               day = c(1,2),
               wtdr2d = T,
               energy_kcal = T,
               total_sugars_g = T,
               both2days = TRUE)
x15<- diag_Pregnant(years = 2007:2017)
x16<- dex_METS.IR(years = 2013, join = "left")
x17<- diag_CKM(years = 2007:2017, component = F)
x00 <- svy_tableone(design = nhs,
                    cv=c('age','energy','bmi','sugar'),
                    gv=c('sex','race','marr','poor','edu','smoke','alcohol','toldht','Hyperlipidemia','CKD','CVD','DM','PA','BMI','CKM'),
                    
                    by='DIGMQ',
                    
                    c_meanPMse = TRUE,
                    g_nSQper = TRUE,
                    g_direction = "v",
                    total = TRUE)
svy_population(design=nhs)
x0$DIGMQ <- quant(x0$DIGM, n = 4,Q = TRUE,round=2)
x0$DIGMQ.median <- quant.median(x0$DIGM, n=4 ,round=3)
x0$newwt <- 1/6*x0$wtsaf4yr
col_rename(x0) <- c('newwt:nhs_wt')
library(mice)
library(VIM)
library(survey)
library(survival)
missValue(c1)
summary(c3)
#类型判断
is.logical(x0$toldht)
is.numeric(x0$PIR)
x0$sugar <- as.numeric(x0$sugar)
#可视化探索
aggr(x0, prop=FALSE, numbers=TRUE)
#####
x0$toldht <- factor(x0$toldht, levels = c('No','Yes'))
x0$Hyperlipidemia <- factor(x0$Hyperlipidemia, levels = c('no','yes'))
x0$CKD <- factor(x0$CKD, levels = c('no','yes'))
x0$marr <- factor(x0$marr, levels = c('Divorced','Married','Never married'))
x0$smoke <- factor(x0$smoke, levels = c('0','1','2'),labels = c('Never','Former','Current'))
x0$alcohol <- factor(x0$alcohol, levels = c('former','heavy','mild','moderate','never'))
x0$edu <- factor(x0$edu, levels = c('Less than high school','High school or equivalent','College or above'))
x0$poor <- factor(x0$poor, levels = c('≤1.00','1.0-3.0','＞3.0'))
ini <- mice(x0,maxit=0)
meth <- ini$method
meth[c('PIR','energy')] <- 'pmm'
meth[c('marr','smoke','alcohol')] <- 'polyreg'
meth['edu'] <- 'polr'
imp <- mice(x0,m=5,method = meth,seed = 123,printFlag = FALSE)
imp$loggedEvents
####检验插补
pdf("插补收敛性图.pdf", width = 12, height = 8)
plot(imp,c('edu','toldht','Hyperlipidemia'))
densityplot(imp,~CKD+energy+marr+edu)
head(complete(imp,action = 3),10)
complete(imp)
c1 <- complete(imp,1)
c2 <- complete(imp,2)
c3 <- complete(imp,3)
c4 <- complete(imp,4)
c5 <- complete(imp,5)
md.pattern(c3)

library(survival)
fit <- svycoxph(Surv(time,status)~DIGM+age+sex+race+marr+edu+PIR+smoke+alcohol+PA+energy,nhs)
summary(fit)
test.ph <- cox.zph(fit)
test.ph

pooled_results <- pool(fit)
summary(pooled_results)
nhs <- svy_design(C1)
f1 <- svycoxph(Surv(time,status)~DIGMQ,nhs)
f2 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,nhs)
f3 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f1,f2,f3,xlsx = 'coxA.xlsx',style = 1) 
f4 <- svycoxph(Surv(time,status)~DIGMQ,
               subset(nhs,cause  %in%  c('heart','no')))
f5 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,
               subset(nhs,cause  %in%  c('heart','no')))
f6 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
               subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f4,f5,f6,xlsx = 'coxB.xlsx',style = 1)
nhs <- svy_design(C2)
f13 <- svycoxph(Surv(time,status)~DIGMQ,nhs)
f14 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,nhs)
f15 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f13,f14,f15,xlsx = 'coxE.xlsx',style = 1) 
f16 <- svycoxph(Surv(time,status)~DIGMQ,
                subset(nhs,cause  %in%  c('heart','no')))
f17 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,
                subset(nhs,cause  %in%  c('heart','no')))
f18 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f16,f17,f18,xlsx = 'coxF.xlsx',style = 1)
nhs <- svy_design(c3)
f25 <- svycoxph(Surv(time,status)~DIGMQ,nhs)
f26 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,nhs)
f27 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f25,f26,f27,xlsx = 'coxI.xlsx',style = 1) 
f28 <- svycoxph(Surv(time,status)~DIGMQ,
                subset(nhs,cause  %in%  c('heart','no')))
f29 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,
                subset(nhs,cause  %in%  c('heart','no')))
f30 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f28,f29,f30,xlsx = 'coxJ.xlsx',style = 1)
nhs <- svy_design(c4)
f37 <- svycoxph(Surv(time,status)~DIGMQ,nhs)
f38 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,nhs)
f39 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f37,f38,f39,xlsx = 'coxM.xlsx',style = 1) 
f40 <- svycoxph(Surv(time,status)~DIGMQ,
                subset(nhs,cause  %in%  c('heart','no')))
f41 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,
                subset(nhs,cause  %in%  c('heart','no')))
f42 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f40,f41,f42,xlsx = 'coxN.xlsx',style = 1)
nhs <- svy_design(C5)
f49 <- svycoxph(Surv(time,status)~DIGMQ,nhs)
f50 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,nhs)
f51 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f49,f50,f51,xlsx = 'coxQ.xlsx',style = 1) 
f52 <- svycoxph(Surv(time,status)~DIGMQ,
                subset(nhs,cause  %in%  c('heart','no')))
f53 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race,
                subset(nhs,cause  %in%  c('heart','no')))
f54 <- svycoxph(Surv(time,status)~DIGMQ+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f52,f53,f54,xlsx = 'coxR.xlsx',style = 1)
x1$digm <- scale(x1$DIGM)
coxph(Surv(time, status) ~ digm_z, data = dat)
f37 <- svycoxph(Surv(time,status)~digm,nhs)
f38 <- svycoxph(Surv(time,status)~digm+age+sex+race,nhs)
f39 <- svycoxph(Surv(time,status)~digm+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,nhs)
crude.Model.n(f37,f38,f39,xlsx = 'coxg.xlsx',style = 1) 
f40 <- svycoxph(Surv(time,status)~digm,
                subset(nhs,cause  %in%  c('heart','no')))
f41 <- svycoxph(Surv(time,status)~digm+age+sex+race,
                subset(nhs,cause  %in%  c('heart','no')))
f42 <- svycoxph(Surv(time,status)~digm+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                subset(nhs,cause  %in%  c('heart','no')))
crude.Model.n(f40,f41,f42,xlsx = 'coxr.xlsx',style = 1)
beta_mat <- do.call(rbind, lapply(fit_list, coef))
vcov_list <- lapply(fit_list, vcov)
m <- length(fit_list)
beta_bar <- colMeans(beta_mat)
W <- Reduce("+", vcov_list) / m
B <- stats::cov(beta_mat)
T_var <- W + (1 + 1 / m) * B
SE <- sqrt(diag(T_var))
W_diag <- diag(W)
B_diag <- diag(B)
df <- (m - 1) * (1 + W_diag / ((1 + 1 / m) * B_diag))^2
df[is.na(df) | is.infinite(df)] <- 999999
t_value <- beta_bar / SE
p_value <- 2 * pt(abs(t_value), df = df, lower.tail = FALSE)
lower <- beta_bar - qt(0.975, df = df) * SE
upper <- beta_bar + qt(0.975, df = df) * SE
result <- data.frame(
    term = names(beta_bar),
    beta = beta_bar,
    SE = SE,
    HR = exp(beta_bar),
    lower_95CI = exp(lower),
    upper_95CI = exp(upper),
    P_value = p_value,
    row.names = NULL)
result <- result %>%
    mutate(
        HR_95CI = paste0(
            sprintf("%.2f", HR),
            " (",
            sprintf("%.2f", lower_95CI),
            "-",
            sprintf("%.2f", upper_95CI),
            ")"),
        P_value_format = ifelse(P_value < 0.001, "<0.001", sprintf("%.3f", P_value)))
return(result)
fit_pool_model <- function(formula, data_list, cvd = FALSE) {
    fit_list <- lapply(data_list, function(dat) {
        if (cvd == TRUE) {
            dat <- dat %>% filter(cause %in% c("heart", "no"))}
        design_i <- make_design(dat)
        svycoxph(formula, design = design_i)})
    pooled_result <- rubin_pool_svycox(fit_list)
    return(pooled_result)}
all_q_model1 <- fit_pool_model(
    Surv(time, status) ~ DIGMQ,
    data_list = imp_list,
    cvd = FALSE)
all_q_model2 <- fit_pool_model(
    Surv(time, status) ~ DIGMQ + age + sex + race,
    data_list = imp_list,
    cvd = FALSE)
all_q_model3 <- fit_pool_model(
    Surv(time, status) ~ DIGMQ + age + sex + race + marr + edu + poor +
        PA + smoke + alcohol + energy,
    data_list = imp_list,
    cvd = FALSE
    cvd_q_model1 <- fit_pool_model(
        Surv(time, status) ~ DIGMQ,
        data_list = imp_list,
        cvd = TRUE)
    cvd_q_model2 <- fit_pool_model(
        Surv(time, status) ~ DIGMQ + age + sex + race,
        data_list = imp_list,
        cvd = TRUE)
    cvd_q_model3 <- fit_pool_model(
        Surv(time, status) ~ DIGMQ + age + sex + race + marr + edu + poor +
            PA + smoke + alcohol + energy,
        data_list = imp_list,
        cvd = TRUE
        all_sd_model1 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd,
            data_list = imp_list,
            cvd = FALSE)
        all_sd_model2 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd + age + sex + race,
            data_list = imp_list,
            cvd = FALSE
        )
        
        all_sd_model3 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd + age + sex + race + marr + edu + poor +
                PA + smoke + alcohol + energy,
            data_list = imp_list,
            cvd = FALSE)
        cvd_sd_model1 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd,
            data_list = imp_list,
            cvd = TRUE)
        cvd_sd_model2 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd + age + sex + race,
            data_list = imp_list,
            cvd = TRUE)
        cvd_sd_model3 <- fit_pool_model(
            Surv(time, status) ~ DIGM_sd + age + sex + race + marr + edu + poor +
                PA + smoke + alcohol + energy,
            data_list = imp_list,
            cvd = TRUE
            all_q_model3
            cvd_q_model3
            all_sd_model3
            cvd_sd_model3
            rite.csv(all_q_model3, "pooled_all_cause_DIGMQ_model3.csv", row.names = FALSE)
            write.csv(cvd_q_model3, "pooled_cvd_DIGMQ_model3.csv", row.names = FALSE)
            write.csv(all_sd_model3, "pooled_all_cause_DIGM_sd_model3.csv", row.names = FALSE)
            write.csv(cvd_sd_model3, "pooled_cvd_DIGM_sd_model3.csv", row.names = FALSE)
            library(rms)
            f0 <- svycoxph(Surv(time,status)~rcs(DIGM),design=nhs)|>reg_table()
            f00 <- svycoxph(Surv(time,status)~rcs(DIGM)+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,design=nhs)
            f01 <- svycoxph(Surv(time,status)~rcs(DIGM)+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                            subset(nhs,cause  %in%  c('heart','no')))
            optimal_nKnots(f00)
            optimal_nKnots(f01)
            f000 <- svycoxph(Surv(time,status)~rcs(DIGM,3)+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,design=nhs)
            f001 <- svycoxph(Surv(time,status)~rcs(DIGM,3)+age+sex+race+marr+edu+poor+PA+smoke+alcohol+energy,
                             design=subset(nhs,cause  %in%  c('heart','no')))
            r000 <- RCS(f000,log=F)
            r001 <- RCS(f001)
            ggplot(r000)
            ggplot(r001)
            getChangepoints(r001)
            library(export)
            p <- ggplot(r001)
            
            s2 <- svykm(Surv(time,status)~DIGMQ,design = nhs)
            svy_kmplot(s2)
            s3 <- svykm(Surv(time,status)~DIGMQ,subset(nhs,cause  %in%  c('heart','no')))
            svy_kmplot(s3)
            svy_kmplot(s2,xlab = 'Follow-up time,months',round = 2,legend.title = 'CDAI quartile',legend.position = c(0.9,0.75))
            svy_kmplot(s2,xlab = 'Time (months)',
                       legend.position = c(0.9,0.75),
                       legend.title = 'Quartile of DI-GM',
                       margin = c(20,1,60,1))
            rt.text.y=c(-0.25,-0.3)
            rt.title.xy = c(500,-0.2)
            rt.text.y = -0.3
            library(export)
            d<- svy_kmplot(s3)
            graph2ppt(d)
            tiff("s3.tiff",width=300,height=380)
            ggsurvplot(s3)
            dev.off()
            
            x0 <- add_col(data = x0,colname = 'age',value = '≤60',condition = x0$age<=60)
            x0 <- add_col(data = x0,colname = 'age',value = '＞60',condition = x0$age>60)
            x0$race <- Recode(x0$race,
                              'Mexican American::other',
                              'Non-Hispanic Black::other') 
            x0$smoke <- Recode(x0$smoke,
                               "0::never smoker",
                               "1::past smoker",
                               '2::current smoker')
            x0$alcohol <- Recode(x0$alcohol,
                                 "former::past drinker",
                                 "heavy::current drinker",
                                 'mild::current drinker',
                                 "moderate::current drinker",
                                 'never::never drinker')
            x0$BMI <- Recode(x0$BMI,
                             "Normal::<30",
                             "Obese::>=30",
                             'Overweight::<30')
            x0$CKD_eGFR <- Recode(x0$CKD_eGFR,
                                  "G1::G12",
                                  "G2::G12",
                                  "G3a::G345",
                                  "G3b::G345",
                                  "G4::G345",
                                  "G5::G345")
            x0$DM <- Recode(x0$DM,
                            "IFG::no",
                            "IGT::no")
            x0 <- add_col(data = x0,colname = 'GFR',value = '<60',condition = x0$eGFR<60)
            x0 <- add_col(data = x0,colname = 'GFR',value = '≥60',condition = x0$eGFR>=60)
            nhs <- svy_design(x0)
            stratum_model(object = subset(nhs,cause  %in%  c('heart','no')),
                          time = 'time',
                          y='status',
                          x='DIGM',
                          stratum = c('age','sex','race','poor','smoke','alcohol','BMI','DM','CKM'),
                          adjust=c('marr','edu','PA','energy'),
                          p=T,
                          xlsx = 's.xlsx',
                          interaction = TRUE,
                          round = 2)
            stratum_model(object = nhs,
                          time = 'time',
                          y='status',
                          x='DIGM',
                          stratum = c('age','sex','race','poor','smoke','alcohol','BMI','DM','CKM'),
                          adjust=c('marr','edu','PA','energy'),
                          p=T,
                          xlsx = 'j.xlsx',
                          interaction = TRUE,
                          round = 2)
            library(readr)
            library(dplyr)
            library(survival)
            library(survey)
            options(survey.lonely.psu = "adjust")
            x0 <- x0 %>%
                mutate(
                    sex = factor(sex),
                    race = factor(race),
                    marr = factor(marr),
                    edu = factor(edu),
                    PIR = factor(PIR),
                    PA = factor(PA),
                    smoke = factor(smoke),
                    alcohol = factor(alcohol),
                    DIGMQ = relevel(factor(DIGMQ), ref = "Q1"),
                    # cause: 0 = censored/alive, 1 = CVD death, 2 = non-CVD death
                    fg_event = factor(
                        cause,
                        levels = c(0, 1, 2),
                        labels = c("censored", "cvd_death", "non_cvd_death")
                    ))
            sd_DIGM <- sd(x0$DIGM, na.rm = TRUE)
            sd_DIGM
            x0$DIGM_sd <- as.numeric(scale(x0$DIGM))
            library(survival)
            library(survey)
            options(survey.lonely.psu = "adjust")
            fg_x0_crude_sd <- finegray(
                Surv(time, fg_event) ~ DIGM_sd + sdmvpsu + sdmvstra + nhs_wt,
                data = x0,
                etype = "cvd_death")
            fg_x0_crude_sd$fg_nhanes_wt <- fg_x0_crude_sd$fgwt * fg_x0_crude_sd$nhs_wt
            fg_design_crude_sd <- svydesign(
                ids = ~sdmvpsu,
                strata = ~sdmvstra,
                weights = ~fg_nhanes_wt,
                nest = TRUE,
                data = fg_x0_crude_sd)
            fit_fg_crude_sd <- svycoxph(
                Surv(fgstart, fgstop, fgstatus) ~ DIGM_sd,
                design = fg_design_crude_sd)
            summary(fit_fg_crude_sd)
            fg_x0_adj_sd <- finegray(
                Surv(time, fg_event) ~ DIGM_sd + age + sex + race + marr + edu + poor +
                    PA + smoke + alcohol + energy +
                    sdmvpsu + sdmvstra + nhs_wt,
                data = x0,
                etype = "cvd_death")
            fg_x0_adj_sd$fg_nhanes_wt <- fg_x0_adj_sd$fgwt * fg_x0_adj_sd$nhs_wt
            fg_design_adj_sd <- svydesign(
                ids = ~sdmvpsu,
                strata = ~sdmvstra,
                weights = ~fg_nhanes_wt,
                nest = TRUE,
                data = fg_x0_adj_sd)
            fit_fg_adj_sd <- svycoxph(
                Surv(fgstart, fgstop, fgstatus) ~ DIGM_sd + age + sex + race + marr +
                    edu + poor + PA + smoke + alcohol + energy,
                design = fg_design_adj_sd)
            summary(fit_fg_adj_sd)
            library(dplyr)
            x0_rate <- x0 %>%
                mutate(
                    followup_years = time / 12,
                    DIGMQ = factor(DIGMQ, levels = c("Q1", "Q2", "Q3", "Q4")),
                    all_death = ifelse(status == 1, 1, 0),
                    cvd_death = ifelse(cause == 1, 1, 0))
            absolute_rate_table <- x0_rate %>%
                group_by(DIGMQ) %>%
                summarise(
                    Participants = n(),
                    Person_years = sum(followup_years, na.rm = TRUE),
                    All_cause_deaths = sum(all_death, na.rm = TRUE),
                    All_cause_death_percent = All_cause_deaths / Participants * 100,
                    All_cause_rate_per_1000_py = All_cause_deaths / Person_years * 1000,
                    
                    CVD_deaths = sum(cvd_death, na.rm = TRUE),
                    CVD_death_percent = CVD_deaths / Participants * 100,
                    CVD_rate_per_1000_py = CVD_deaths / Person_years * 1000,
                    
                    Weighted_all_cause_rate_per_1000_py =
                        sum(nhs_wt * all_death, na.rm = TRUE) /
                        sum(nhs_wt * followup_years, na.rm = TRUE) * 1000,
                    
                    Weighted_CVD_rate_per_1000_py =
                        sum(nhs_wt * cvd_death, na.rm = TRUE) /
                        sum(nhs_wt * followup_years, na.rm = TRUE) * 1000) %>%
                absolute_rate_table <- absolute_rate_table %>%
                mutate(
                    All_cause_rate_difference_vs_Q1 =
                        All_cause_rate_per_1000_py - All_cause_rate_per_1000_py[DIGMQ == "Q1"],
                    
                    CVD_rate_difference_vs_Q1 =
                        CVD_rate_per_1000_py - CVD_rate_per_1000_py[DIGMQ == "Q1"],
                    
                    Weighted_all_cause_rate_difference_vs_Q1 =
                        Weighted_all_cause_rate_per_1000_py - 
                        Weighted_all_cause_rate_per_1000_py[DIGMQ == "Q1"],
                    
                    Weighted_CVD_rate_difference_vs_Q1 =
                        Weighted_CVD_rate_per_1000_py - 
                        Weighted_CVD_rate_per_1000_py[DIGMQ == "Q1"])
            absolute_rate_table_final <- absolute_rate_table %>%
                mutate(
                    across(
                        c(
                            Person_years,
                            All_cause_death_percent,
                            All_cause_rate_per_1000_py,
                            CVD_death_percent,
                            CVD_rate_per_1000_py,
                            Weighted_all_cause_rate_per_1000_py,
                            Weighted_CVD_rate_per_1000_py,
                            All_cause_rate_difference_vs_Q1,
                            CVD_rate_difference_vs_Q1,
                            Weighted_all_cause_rate_difference_vs_Q1,
                            Weighted_CVD_rate_difference_vs_Q1
                        ),
                        ~ round(.x, 2)))
            absolute_rate_table_final
            write.csv(
                absolute_rate_table_final,
                "absolute_mortality_rates_by_DIGM_quartiles.csv",
                row.names = FALSE)
            x1 <- drug_anti.Diabetic(
                data=x0,
                years = 2007:2017,
                take_drug = TRUE,
                DrugNumber = FALSE,
                drugname = FALSE,
                fdaNDC = FALSE,
                dcn = FALSE,
                icn = FALSE,
                icd10 = FALSE,
                yes.code = NULL,
                other.code = NULL,
                no.code = NULL,
                remove.other = TRUE,
                dup.take.drug = "remove",
                join = "left",
                Year = FALSE)
            x2 <- drug_anti.Hypertensive(
                data=x1,
                years = 2007:2017,
                take_drug = TRUE,
                DrugNumber = FALSE,
                drugname = FALSE,
                fdaNDC = FALSE,
                dcn = FALSE,
                icn = FALSE,
                icd10 = FALSE,
                yes.code = NULL,
                other.code = NULL,
                no.code = NULL,
                remove.other = TRUE,
                dup.take.drug = "remove",
                join = "left",
                Year = FALSE)
            x3 <- drug_anti.Hyperlipidemic(
                data=x2,
                years = 2007:2017,
                take_drug = TRUE,
                DrugNumber = FALSE,
                drugname = FALSE,
                fdaNDC = FALSE,
                dcn = FALSE,
                icn = FALSE,
                icd10 = FALSE,
                yes.code = NULL,
                other.code = NULL,
                no.code = NULL,
                remove.other = TRUE,
                dup.take.drug = "remove",
                join = "left",
                Year = FALSE)
            col_rename(x3) <- c('take_drug.x:drugDM','take_drug.y:drugHP','take_drug:drugHL')
            x3$drugHL<- Recode(x3$drugHL,
                               'no::0',
                               'other::0',
                               'yes::1',
                               to.numeric=FALSE)
            x3$drug = paste0(x3$drugDM,'~', x3$drugHP,'~', x3$drugHL,'~')
            x3$drug <- Recode(x3$drug,
                              "0~0~0~::no", 
                              "0~0~1~::Yes", 
                              "0~1~1~::Yes", 
                              "0~1~0~::Yes", 
                              "1~0~0~::Yes", 
                              "1~0~1~::Yes", 
                              "1~1~1~::Yes", 
                              "1~1~0~::Yes", 
                              "NA~NA~NA~::no", 
                              to.numeric = FALSE)