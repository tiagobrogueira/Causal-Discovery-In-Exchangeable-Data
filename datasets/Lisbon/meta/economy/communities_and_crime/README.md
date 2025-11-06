# Source: communities_and_crime
**Category:** economy
**README generated:** 2025-10-02 12:52:18Z

## Source URL(s)
- https://archive.ics.uci.edu/dataset/183/communities+and+crime

## Citation(s)
- Redmond, M. (2002). Communities and Crime [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53W3X.

## Variables (X → Y)
- Demographics(agePct12t21,agePct12t29,agePct16t24,MalePctDivorce,MalePctNevMarr,FemalePctDiv,PctTeen2Par,PctKids2Par,PctYoungKids2Par,PctFam2Par,PersPerOwnOccHous,PersPerRentOccHous) → ViolentCrimesPerPop
- RaceEthnicity(racePctWhite,racepctblack,racePctHisp,blackPerCap,whitePerCap,AsianPerCap,HispPerCap,OtherPerCap,PctSpeakEnglOnly,PctNotSpeakEnglWell) → ViolentCrimesPerPop
- Socioeconomics(medIncome,medFamInc,perCapInc,PctUnemployed,PctEmploy,PctOccupManu,PctOccupMgmtProf,pctWWage,pctWInvInc,pctWPubAsst,PctNotHSGrad,PctLess9thGrade,PctBSorMore) → ViolentCrimesPerPop
- FamilyStability(TotalPctDiv,MalePctDivorce,MalePctNevMarr,PctKids2Par,PctTeen2Par,PctYoungKids2Par,PctFam2Par) → ViolentCrimesPerPop
- Housing(HousVacant,PctHousOccup,PctHousOwnOcc,PctHousLess3BR,MedRent,MedRentPctHousInc,RentLowQ,RentMedian,RentHighQ,OwnOccHiQuart,OwnOccMedVal,OwnOccLowQuart,PctHousNoPhone,PctWOFullPlumb,PctVacantBoarded) → ViolentCrimesPerPop
- PovertyInequality(PctPopUnderPov,NumUnderPov,PctIlleg,NumIlleg,PctPersOwnOccup,PctPersDenseHous) → ViolentCrimesPerPop
- ImmigrationMobility(PctRecentImmig,PctRecImmig5,PctRecImmig8,PctRecImmig10,PctImmigRecent,PctImmigRec5,PctImmigRec8,PctImmigRec10,NumImmig,PctSameHouse85) → ViolentCrimesPerPop
- Policing(LemasSwornFT,LemasSwFTPerPop,LemasSwFTFieldOps,LemasSwFTFieldPerPop,LemasTotalReq,LemasTotReqPerPop,PolicPerPop,PolicBudgPerPop,PolicOperBudg,PolicCars,PolicReqPerOffic,OfficAssgnDrugUnits,NumStreet,NumInShelters,PctPolicWhite,PctPolicBlack,PctPolicMinor,RacialMatchCommPol) → ViolentCrimesPerPop
- CommunityUrbanization(PopDens,population,numbUrban,state) → ViolentCrimesPerPop

## Causal reasoning

Community-level demographic, socioeconomic, housing, family, policing, and urbanization factors causally influence violent crime rates by shaping economic opportunity, social stability, population density, and law enforcement capacity.
