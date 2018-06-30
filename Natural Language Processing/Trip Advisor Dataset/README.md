# Summary

This dataset contains 
			```
			1. Text_Reviews_R.csv - Reviews(And also the overall rating) being scraped from Trip advisor website on various hotels and restaurants. Scraping was done semi automatically.
			2. Review_SentimentAnalysis.R - Sentiment for each reviews are analysed from frequent keywords
			```
Reviews were mapped to the overall rating. Score of 2/5 and 1/5 were assumed to be negative sentiment 
and 4/5 and 5/5 were assumed to be positive sentiment. Keywords which were most contributing for the 
negative sentiment and positive sentiment were found out. Decision tree is generated as part of the sentiment
analysis, which shows how significant a keyword's contribution towards positive sentiment and negative sentiment. 
When the number of reviews are too much to conduct manual analysis, we could figure out keywords based on its 
significance of contribution to the positive and negative sentiment towards the hotel 


			