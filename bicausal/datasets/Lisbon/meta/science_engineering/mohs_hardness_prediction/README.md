# Source: mohs_hardness_prediction
**Category:** science_engineering
**README generated:** 2025-10-02 12:52:18Z

## Source URL(s)
- https://www.kaggle.com/datasets/jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning

## Citation(s)
- Jocelyn Dumlao. Prediction of Mohs Hardness with Machine Learning [Dataset]. Kaggle. https://www.kaggle.com/datasets/jocelyndumlao/prediction-of-mohs-hardness-with-machine-learning

## Variables (X → Y)
- R_cov_element_Average → Hardness
- R_cov_element_Average → density_Average
- R_vdw_element_Average → Hardness
- R_vdw_element_Average → density_Average
- allelectrons_Average → Hardness
- allelectrons_Average → R_cov_element_Average
- allelectrons_Average → R_vdw_element_Average
- allelectrons(allelectrons_Average,allelectrons_Total) → density(density_Average,density_Total)
- allelectrons_Average → el_neg_chi_Average
- allelectrons_Average → ionenergy_Average
- atomicweight_Average → Hardness
- atomicweight_Average → R_cov_element_Average
- atomicweight_Average → R_vdw_element_Average
- atomicweight_Average → density(density_Average,density_Total)
- atomicweight_Average → el_neg_chi_Average
- atomicweight_Average → ionenergy_Average
- el_neg_chi_Average → Hardness
- el_neg_chi_Average → density_Average
- ionenergy_Average → Hardness
- ionenergy_Average → density_Average
- val_e_Average → Hardness
- val_e_Average → R_cov_element_Average
- val_e_Average → R_vdw_element_Average
- val_e_Average → density_Average
- val_e_Average → el_neg_chi_Average
- val_e_Average → ionenergy_Average

## Causal reasoning

This dataset contains 4 different steps of a causal ladder. The most basic features (atomicweight_Average, allelectrons_Total, allelectrons_Average, val_e_Average) are basic atom properties. In turn, they interact with other atoms inside the material causing the bonding variables (ionenergy_Average,el_neg_chi_Average,R_vdw_element_Average, R_cov_element_Average). All the previous variables leads to different structural packing (density_Total, density_Average) and hardness of the material.
