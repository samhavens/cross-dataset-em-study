# Leaderboard

a quick-and-dirty markdown dump of the *per-dataset* cross-dataset-EM leaderboards extracted from **Table 3** of the [EDBT 2025 paper “a deep dive into cross-dataset entity matching with large and small language models.”](https://openproceedings.org/2025/conf/edbt/paper-224.pdf)  F1 scores are macro-averaged; I keep only the top-5 for brevity.

> **caveats**
>
> * jellyfish rows are *italicised* ⇢ the paper flags them as *seen-during-training* and therefore **not strictly cross-dataset**.
> * std-devs are omitted (they’re ± ≤10 in all cases).
> * best score per dataset is **bold**.
> * table & values from [EDBT 2025 paper “a deep dive into cross-dataset entity matching with large and small language models.”]

---

### abt

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | **matchgpt \[gpt-4]**    | **92.4** |
| 2 | anymatch \[llama3.2]     | 89.3     |
| 3 | unicorn                  | 87.8     |
| 4 | matchgpt \[gpt-4o-mini]  | 87.2     |
| 5 | matchgpt \[mixtral-8×7b] | 80.7     |

### wdc

| # | model                     | F1       |
| - | ------------------------- | -------- |
| 1 | **matchgpt \[gpt-4]**     | **89.1** |
| 2 | matchgpt \[gpt-4o-mini]   | 88.4     |
| 3 | matchgpt \[gpt-3.5-turbo] | 81.9     |
| 4 | matchgpt \[beluga2]       | 78.6     |
| 5 | matchgpt \[solar]         | 76.6     |

### dblp\_acm (dbac)

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | *jellyfish*              | *97.7*   |
| 2 | **anymatch \[llama3.2]** | **96.5** |
| 3 | anymatch \[t5]           | 96.4     |
| 4 | matchgpt \[gpt-4]        | 96.0     |
| 5 | anymatch \[gpt-2]        | 95.2     |

### dblp\_scholar (dbgo)

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | *jellyfish*              | *93.4*   |
| 2 | **anymatch \[llama3.2]** | **89.8** |
| 3 | matchgpt \[gpt-4]        | 87.9     |
| 4 | matchgpt \[gpt-4o-mini]  | 87.4     |
| 5 | unicorn                  | 86.4     |

### fodors\_zagat (foza)

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | **anymatch \[llama3.2]** | **99.6** |
| 2 | *jellyfish*              | *97.3*   |
| 3 | anymatch \[gpt-2]        | 96.4     |
| 4 | anymatch \[t5]           | 95.4     |
| 5 | matchgpt \[gpt-4]        | 95.1     |

### zomato\_yelp (zoye)

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | *jellyfish*              | *99.1*   |
| 2 | **anymatch \[llama3.2]** | **98.2** |
| 3 | matchgpt \[gpt-4o-mini]  | 98.1     |
| 4 | matchgpt \[gpt-4]        | 97.9     |
| 5 | matchgpt \[solar]        | 97.1     |

### amazon\_google (amgo)

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | **matchgpt \[gpt-4]** | **75.0** |
| 2 | anymatch \[llama3.2]      | 69.3     |
| 3 | anymatch \[t5]           | 64.4     |
| 4 | unicorn                  | 64.0     |
| 5 | matchgpt \[gpt-4o-mini]  | 60.7     |

### beer

| # | model                    | F1       |
| - | ------------------------ | -------- |
| 1 | **anymatch \[llama3.2]** | **95.3** |
| 2 | anymatch \[gpt-2]        | 91.2     |
| 3 | *jellyfish*              | *90.1*   |
| 4 | anymatch \[t5]           | 89.2     |
| 5 | ditto                    | 89.1     |

### itunes\_amazon (itam)

| # | model                   | F1       |
| - | ----------------------- | -------- |
| 1 | **anymatch \[gpt-2]**   | **85.0** |
| 2 | anymatch \[llama3.2]    | 82.3     |
| 3 | anymatch \[t5]          | 79.6     |
| 4 | matchgpt \[gpt-4o-mini] | 69.6     |
| 5 | matchgpt \[solar]       | 67.3     |

### rotten\_imdb (roim)

| # | model                   | F1       |
| - | ----------------------- | -------- |
| 1 | **matchgpt \[gpt-4]**   | **97.2** |
| 2 | *jellyfish*             | *97.0*   |
| 3 | anymatch \[llama3.2]    | 95.9     |
| 4 | matchgpt \[gpt-4o-mini] | 95.7     |
| 5 | matchgpt \[beluga2]     | 90.8     |

### walmart\_amazon (waam)

| # | model                   | F1       |
| - | ----------------------- | -------- |
| 1 | **matchgpt \[gpt-4]**   | **85.1** |
| 2 | matchgpt \[gpt-4o-mini] | 82.9     |
| 3 | *jellyfish*             | *81.4*   |
| 4 | anymatch \[llama3.2]    | 77.2     |
| 5 | matchgpt \[beluga2]     | 77.1     |


