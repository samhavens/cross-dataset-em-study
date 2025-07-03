# Dataset Sizes and Statistics

This document provides a comprehensive overview of all entity matching datasets in this repository, including record counts, token estimates, and pair statistics.

## Overview

| Dataset | Table A Records | Table B Records | Total Records | Avg Tokens/Record A | Avg Tokens/Record B |
|---------|----------------|----------------|---------------|-------------------|-------------------|
| **abt_buy** | 1,081 | 1,092 | 2,173 | 59.0 | 20.1 |
| **amazon_google** | 1,363 | 3,226 | 4,589 | 11.3 | 13.2 |
| **beer** | 4,345 | 3,000 | 7,345 | 17.6 | 16.3 |
| **dblp_acm** | 2,616 | 2,294 | 4,910 | 25.6 | 30.4 |
| **dblp_scholar** | 2,616 | 64,263 | 66,879 | 23.8 | 25.7 |
| **fodors_zagat** | 533 | 331 | 864 | 24.6 | 22.6 |
| **itunes_amazon** | 6,907 | 55,923 | 62,830 | 48.8 | 39.9 |
| **rotten_imdb** | 7,390 | 6,407 | 13,797 | 15.4 | 19.3 |
| **walmart_amazon** | 2,554 | 22,074 | 24,628 | 19.0 | 21.3 |
| **zomato_yelp** | 6,960 | 3,897 | 10,857 | 17.7 | 16.5 |

## Test Set Statistics

| Dataset | Test Pairs | Positive Pairs | Negative Pairs | Positive Rate |
|---------|------------|----------------|----------------|---------------|
| **abt_buy** | 1,916 | 206 | 1,710 | 10.8% |
| **amazon_google** | 2,293 | 234 | 2,059 | 10.2% |
| **beer** | 91 | 14 | 77 | 15.4% |
| **dblp_acm** | 2,473 | 444 | 2,029 | 18.0% |
| **dblp_scholar** | 5,742 | 1,070 | 4,672 | 18.6% |
| **fodors_zagat** | 189 | 22 | 167 | 11.6% |
| **itunes_amazon** | 109 | 27 | 82 | 24.8% |
| **rotten_imdb** | 120 | 38 | 82 | 31.7% |
| **walmart_amazon** | 2,049 | 193 | 1,856 | 9.4% |
| **zomato_yelp** | 89 | 21 | 68 | 23.6% |

## Validation Set Statistics

| Dataset | Valid Pairs | Positive Pairs | Negative Pairs | Positive Rate |
|---------|-------------|----------------|----------------|---------------|
| **abt_buy** | 1,916 | 206 | 1,710 | 10.8% |
| **amazon_google** | 2,293 | 234 | 2,059 | 10.2% |
| **beer** | 91 | 14 | 77 | 15.4% |
| **dblp_acm** | 2,473 | 444 | 2,029 | 18.0% |
| **dblp_scholar** | 5,742 | 1,070 | 4,672 | 18.6% |
| **fodors_zagat** | 190 | 22 | 168 | 11.6% |
| **itunes_amazon** | 109 | 27 | 82 | 24.8% |
| **rotten_imdb** | 120 | 42 | 78 | 35.0% |
| **walmart_amazon** | 2,049 | 193 | 1,856 | 9.4% |
| **zomato_yelp** | 89 | 10 | 79 | 11.2% |

## Training Set Statistics

| Dataset | Train Pairs | Positive Pairs | Negative Pairs | Positive Rate |
|---------|-------------|----------------|----------------|---------------|
| **abt_buy** | 5,743 | 616 | 5,127 | 10.7% |
| **amazon_google** | 6,874 | 699 | 6,175 | 10.2% |
| **beer** | 268 | 40 | 228 | 14.9% |
| **dblp_acm** | 7,417 | 1,332 | 6,085 | 18.0% |
| **dblp_scholar** | 17,223 | 3,207 | 14,016 | 18.6% |
| **fodors_zagat** | 567 | 66 | 501 | 11.6% |
| **itunes_amazon** | 321 | 78 | 243 | 24.3% |
| **rotten_imdb** | 360 | 110 | 250 | 30.6% |
| **walmart_amazon** | 6,144 | 576 | 5,568 | 9.4% |
| **zomato_yelp** | 266 | 59 | 207 | 22.2% |

## Token Analysis

### Total Token Counts by Dataset

| Dataset | Table A Tokens | Table B Tokens | Total Tokens |
|---------|----------------|----------------|--------------|
| **abt_buy** | 63,831 | 21,915 | 85,746 |
| **amazon_google** | 15,386 | 42,740 | 58,126 |
| **beer** | 76,537 | 48,790 | 125,327 |
| **dblp_acm** | 66,896 | 69,843 | 136,739 |
| **dblp_scholar** | 62,179 | 1,649,252 | 1,711,431 |
| **fodors_zagat** | 13,106 | 7,480 | 20,586 |
| **itunes_amazon** | 336,802 | 2,232,682 | 2,569,484 |
| **rotten_imdb** | 113,999 | 123,624 | 237,623 |
| **walmart_amazon** | 48,590 | 470,241 | 518,831 |
| **zomato_yelp** | 123,090 | 64,296 | 187,386 |

## Dataset Characteristics

### Small Datasets (< 1,000 test pairs)
- **beer**: 91 test pairs, 7,345 total records
- **zomato_yelp**: 89 test pairs, 10,857 total records  
- **itunes_amazon**: 109 test pairs, 62,830 total records
- **rotten_imdb**: 120 test pairs, 13,797 total records
- **fodors_zagat**: 189 test pairs, 864 total records

### Medium Datasets (1,000-3,000 test pairs)
- **abt_buy**: 1,916 test pairs, 2,173 total records
- **amazon_google**: 2,293 test pairs, 4,589 total records
- **walmart_amazon**: 2,049 test pairs, 24,628 total records
- **dblp_acm**: 2,473 test pairs, 4,910 total records

### Large Datasets (> 3,000 test pairs)
- **dblp_scholar**: 5,742 test pairs, 66,879 total records

### High Token Density Datasets
- **abt_buy**: 59.0 avg tokens per record (Table A)
- **itunes_amazon**: 48.8 avg tokens per record (Table A)
- **dblp_acm**: 25.6-30.4 avg tokens per record

### Imbalanced Datasets (< 15% positive pairs)
- **walmart_amazon**: 9.4% positive rate
- **amazon_google**: 10.2% positive rate  
- **abt_buy**: 10.8% positive rate
- **fodors_zagat**: 11.6% positive rate

### Balanced Datasets (> 20% positive pairs)
- **itunes_amazon**: 24.8% positive rate
- **rotten_imdb**: 31.7% positive rate
- **zomato_yelp**: 23.6% positive rate

## Notes

- Token estimates are calculated using a simple word-count heuristic (words × 1.3)
- Test and validation sets appear to be identical for most datasets
- The **wdc** dataset appears to be incomplete or missing files
- **dblp_scholar** and **itunes_amazon** are the largest datasets by total token count
- Most datasets are heavily imbalanced toward negative pairs (non-matches) 