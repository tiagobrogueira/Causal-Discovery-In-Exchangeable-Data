# Source: online_news_popularity
**Category:** human predictions
**README generated:** 2025-10-02 12:52:18Z

## Source URL(s)
- https://archive.ics.uci.edu/dataset/332/online+news+popularity

## Citation(s)
- Fernandes, K., Vinagre, P., Cortez, P., & Sernadela, P. (2015). Online News Popularity [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5NS3V.

## Variables (X → Y)
[Filtered by: data_channel_is_tech, weekday_is_friday, weekday_is_saturday, weekday_is_thursday, weekday_is_wednesday, data_channel_is_entertainment, weekday_is_tuesday, data_channel_is_socmed, data_channel_is_bus, weekday_is_monday, data_channel_is_lifestyle, is_weekend, data_channel_is_world, weekday_is_sunday]

- lad analysis(LDA_00,LDA_02) → shares
- min_positive_polarity → shares
- article metrics(n_tokens_content,n_unique_tokens) → shares
- content popularity metrics(kw_avg_avg,kw_max_avg,self_reference_avg_sharess,self_reference_min_shares,self_reference_max_shares) → shares

## Causal reasoning

In this dataset, the effect is the popularity of the articles based on its linguistic and semantic features, as well as the prior popularity of the topic in question.