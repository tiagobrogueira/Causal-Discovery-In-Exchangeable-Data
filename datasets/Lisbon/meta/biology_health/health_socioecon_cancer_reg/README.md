# Source: health_socioecon_cancer_reg
**Category:** biology_health
**README generated:** 2025-10-02 12:52:18Z

## Source URL(s)
- https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioec

## Citation(s)
- TheDevastator. (2024). Uncovering Trends in Health Outcomes and Socioeconomic Factors [Dataset]. Kaggle. https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioec

## Variables (X → Y)
- age(medianage,medianagefemale,medianagemale) → absolute fatalities(avganncount,avgdeathsperyear)
- medincome → absolute fatalities(avganncount,avgdeathsperyear)
- medincome → target_deathrate
- race distribution(pctasian,pctblack,pctwhite,pctotherrace) → absolute fatalities(avganncount,avgdeathsperyear)
- race distribution(pctasian,pctblack) → target_deathrate
- degree of instruction(pctbachdeg18_24,pctbachdeg25_over,pcths18_24,pcths25_over,pctnohs18_24,pctsomecol18_24) → absolute fatalities(avganncount,avgdeathsperyear)
- degree of instruction(pctbachdeg18_24,pctbachdeg25_over,pcths18_24,pcths25_over,pctsomecol18_24) → target_deathrate
- employment status(pctemployed16_over,pctunemployed16_over) → absolute fatalities(avganncount,avgdeathsperyear)
- employment status(pctemployed16_over,pctunemployed16_over) → target_deathrate
- percentmarried → avgdeathsperyear
- marriage status(pctmarriedhouseholds,percentmarried) → target_deathrate
- coverage(pctempprivcoverage,pctprivatecoverage,pctprivatecoveragealone,pctpubliccoverage,pctpubliccoveragealone) → absolute fatalities(avganncount,avgdeathsperyear)
- coverage(pctempprivcoverage,pctprivatecoverage,pctprivatecoveragealone,pctpubliccoverage,pctpubliccoveragealone) → target_deathrate
- povertypercent → avganncount
- povertypercent → target_deathrate
- studypercap → absolute fatalities(avganncount,avgdeathsperyear)

## Causal reasoning

In this dataset, the causes are different features of a population, such as the medical coverage and the degree of education that can influence the deaths in the given population.