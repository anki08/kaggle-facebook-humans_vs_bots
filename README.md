# kaggle-facebook-humans_vs_bots
The link to the question is  https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot
The bids file could not be uploaded due to its large size
I thought that the main features that could distinguish a bot and a human were
  1) the time taken for response(bots should be faster)
  2) mean and median time for response(for bots it should be the same not much deviation)
  3) bot could apply from different countries and ip so found their cost
  4) the range of time for which the bidder participated 
  5) some merchandise might have more bot participation

I found out after analysis that the ip_count , country count gave nearly the same result so we could remove them

I used XGBoost algo for the final prediction
