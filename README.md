# amazon-review-classifier
Using Ensemble Learning to select high quality Amazon reviews.

## Data Fields
- Id: identify test cases (only appear in test set)
- reviewerID: unique id of each reviwer
- asin: unique id of each item
- reviewText: content of review in English, without preprocessing
- overall: the rating user gives to item (from 1 to 5)
- votes_up: number of up votes to this review (only appear in training set)
- votes_all: number of total votes to this review (only appear in training set)
- label: 0 for low quality, and 1 for high quality (only appear in training set)

Reviews with votes_up / votes_all ≥ 0.9 are considered as high quality reviews. All the reviews are assured to have at least 5 votes_all.
