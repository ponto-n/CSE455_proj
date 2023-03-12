---
license: apache-2.0
tags:
- image-classification
- vision
- generated_from_trainer
model-index:
- name: run005
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# run005

This model is a fine-tuned version of [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1793
- Mse: 0.1793
- Mae: 0.2732

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 100.0

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Mse    | Mae    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:------:|
| 2.3333        | 1.0   | 603   | 1.8802          | 1.8802 | 1.0889 |
| 1.0742        | 2.0   | 1206  | 0.9201          | 0.9201 | 0.6886 |
| 0.851         | 3.0   | 1809  | 0.6988          | 0.6988 | 0.5804 |
| 0.7477        | 4.0   | 2412  | 0.6316          | 0.6316 | 0.5609 |
| 0.6662        | 5.0   | 3015  | 0.5671          | 0.5671 | 0.5189 |
| 0.6553        | 6.0   | 3618  | 0.5791          | 0.5791 | 0.5224 |
| 0.5698        | 7.0   | 4221  | 0.4736          | 0.4736 | 0.4714 |
| 0.5482        | 8.0   | 4824  | 0.4933          | 0.4933 | 0.4817 |
| 0.5437        | 9.0   | 5427  | 0.4930          | 0.4930 | 0.4815 |
| 0.505         | 10.0  | 6030  | 0.4386          | 0.4386 | 0.4426 |
| 0.4738        | 11.0  | 6633  | 0.4497          | 0.4497 | 0.4472 |
| 0.4741        | 12.0  | 7236  | 0.3986          | 0.3986 | 0.4233 |
| 0.4385        | 13.0  | 7839  | 0.3703          | 0.3703 | 0.4164 |
| 0.45          | 14.0  | 8442  | 0.3363          | 0.3363 | 0.3917 |
| 0.4349        | 15.0  | 9045  | 0.3745          | 0.3745 | 0.4161 |
| 0.4222        | 16.0  | 9648  | 0.3842          | 0.3842 | 0.4086 |
| 0.4164        | 17.0  | 10251 | 0.3665          | 0.3665 | 0.4061 |
| 0.3865        | 18.0  | 10854 | 0.2860          | 0.2860 | 0.3632 |
| 0.3693        | 19.0  | 11457 | 0.3397          | 0.3397 | 0.3810 |
| 0.3546        | 20.0  | 12060 | 0.3433          | 0.3433 | 0.3841 |
| 0.3748        | 21.0  | 12663 | 0.3383          | 0.3383 | 0.3882 |
| 0.3645        | 22.0  | 13266 | 0.3271          | 0.3271 | 0.3753 |
| 0.3503        | 23.0  | 13869 | 0.3182          | 0.3182 | 0.3671 |
| 0.3578        | 24.0  | 14472 | 0.3257          | 0.3257 | 0.3690 |
| 0.3458        | 25.0  | 15075 | 0.3140          | 0.3140 | 0.3689 |
| 0.335         | 26.0  | 15678 | 0.2605          | 0.2605 | 0.3448 |
| 0.302         | 27.0  | 16281 | 0.2925          | 0.2925 | 0.3523 |
| 0.3242        | 28.0  | 16884 | 0.2807          | 0.2807 | 0.3523 |
| 0.3144        | 29.0  | 17487 | 0.3176          | 0.3176 | 0.3580 |
| 0.3           | 30.0  | 18090 | 0.2562          | 0.2562 | 0.3407 |
| 0.3148        | 31.0  | 18693 | 0.2745          | 0.2745 | 0.3472 |
| 0.2934        | 32.0  | 19296 | 0.2968          | 0.2968 | 0.3517 |
| 0.3064        | 33.0  | 19899 | 0.2903          | 0.2903 | 0.3549 |
| 0.2946        | 34.0  | 20502 | 0.2684          | 0.2684 | 0.3363 |
| 0.3007        | 35.0  | 21105 | 0.2490          | 0.2490 | 0.3354 |
| 0.3112        | 36.0  | 21708 | 0.2628          | 0.2628 | 0.3376 |
| 0.2885        | 37.0  | 22311 | 0.2924          | 0.2924 | 0.3465 |
| 0.3016        | 38.0  | 22914 | 0.2857          | 0.2857 | 0.3374 |
| 0.2691        | 39.0  | 23517 | 0.2691          | 0.2691 | 0.3281 |
| 0.274         | 40.0  | 24120 | 0.2685          | 0.2685 | 0.3291 |
| 0.2773        | 41.0  | 24723 | 0.2686          | 0.2686 | 0.3230 |
| 0.2822        | 42.0  | 25326 | 0.2865          | 0.2865 | 0.3343 |
| 0.2659        | 43.0  | 25929 | 0.2535          | 0.2535 | 0.3208 |
| 0.2766        | 44.0  | 26532 | 0.2625          | 0.2625 | 0.3244 |
| 0.2758        | 45.0  | 27135 | 0.2150          | 0.2150 | 0.3068 |
| 0.2605        | 46.0  | 27738 | 0.2527          | 0.2527 | 0.3193 |
| 0.2426        | 47.0  | 28341 | 0.2547          | 0.2547 | 0.3250 |
| 0.256         | 48.0  | 28944 | 0.2189          | 0.2189 | 0.3081 |
| 0.2653        | 49.0  | 29547 | 0.2186          | 0.2186 | 0.3119 |
| 0.2494        | 50.0  | 30150 | 0.2544          | 0.2544 | 0.3162 |
| 0.256         | 51.0  | 30753 | 0.2365          | 0.2365 | 0.3145 |
| 0.2422        | 52.0  | 31356 | 0.2273          | 0.2273 | 0.3081 |
| 0.2429        | 53.0  | 31959 | 0.2136          | 0.2136 | 0.2969 |
| 0.2612        | 54.0  | 32562 | 0.2217          | 0.2217 | 0.3088 |
| 0.2507        | 55.0  | 33165 | 0.2683          | 0.2683 | 0.3187 |
| 0.2464        | 56.0  | 33768 | 0.2112          | 0.2112 | 0.2965 |
| 0.2443        | 57.0  | 34371 | 0.2166          | 0.2166 | 0.3000 |
| 0.2317        | 58.0  | 34974 | 0.2038          | 0.2038 | 0.2931 |
| 0.2427        | 59.0  | 35577 | 0.2133          | 0.2133 | 0.2976 |
| 0.2422        | 60.0  | 36180 | 0.2169          | 0.2169 | 0.2991 |
| 0.2533        | 61.0  | 36783 | 0.2383          | 0.2383 | 0.3158 |
| 0.2297        | 62.0  | 37386 | 0.1892          | 0.1892 | 0.2901 |
| 0.2429        | 63.0  | 37989 | 0.2321          | 0.2321 | 0.2991 |
| 0.2262        | 64.0  | 38592 | 0.2311          | 0.2311 | 0.3060 |
| 0.2178        | 65.0  | 39195 | 0.2244          | 0.2244 | 0.3063 |
| 0.2217        | 66.0  | 39798 | 0.2165          | 0.2165 | 0.2983 |
| 0.2398        | 67.0  | 40401 | 0.2297          | 0.2297 | 0.3076 |
| 0.2242        | 68.0  | 41004 | 0.2190          | 0.2190 | 0.3004 |
| 0.2327        | 69.0  | 41607 | 0.1808          | 0.1808 | 0.2796 |
| 0.218         | 70.0  | 42210 | 0.1997          | 0.1997 | 0.2886 |
| 0.2324        | 71.0  | 42813 | 0.2077          | 0.2077 | 0.2995 |
| 0.228         | 72.0  | 43416 | 0.2250          | 0.2250 | 0.3083 |
| 0.2247        | 73.0  | 44019 | 0.2069          | 0.2069 | 0.2933 |
| 0.2139        | 74.0  | 44622 | 0.1985          | 0.1985 | 0.2893 |
| 0.2275        | 75.0  | 45225 | 0.2181          | 0.2181 | 0.3030 |
| 0.2144        | 76.0  | 45828 | 0.2023          | 0.2023 | 0.2915 |
| 0.1977        | 77.0  | 46431 | 0.2311          | 0.2311 | 0.3084 |
| 0.2182        | 78.0  | 47034 | 0.2138          | 0.2138 | 0.2931 |
| 0.2198        | 79.0  | 47637 | 0.2096          | 0.2096 | 0.2959 |
| 0.2221        | 80.0  | 48240 | 0.2119          | 0.2119 | 0.2951 |
| 0.2128        | 81.0  | 48843 | 0.1850          | 0.1850 | 0.2797 |
| 0.2313        | 82.0  | 49446 | 0.1986          | 0.1986 | 0.2874 |
| 0.2112        | 83.0  | 50049 | 0.2342          | 0.2342 | 0.3075 |
| 0.1991        | 84.0  | 50652 | 0.2177          | 0.2177 | 0.2946 |
| 0.207         | 85.0  | 51255 | 0.1874          | 0.1874 | 0.2851 |
| 0.1985        | 86.0  | 51858 | 0.1800          | 0.1800 | 0.2792 |
| 0.21          | 87.0  | 52461 | 0.2108          | 0.2108 | 0.2889 |
| 0.215         | 88.0  | 53064 | 0.1745          | 0.1745 | 0.2768 |
| 0.1968        | 89.0  | 53667 | 0.1828          | 0.1828 | 0.2815 |
| 0.2262        | 90.0  | 54270 | 0.2025          | 0.2025 | 0.2898 |
| 0.2136        | 91.0  | 54873 | 0.2089          | 0.2089 | 0.2949 |
| 0.2079        | 92.0  | 55476 | 0.1679          | 0.1679 | 0.2732 |
| 0.2042        | 93.0  | 56079 | 0.1882          | 0.1882 | 0.2838 |
| 0.2036        | 94.0  | 56682 | 0.2156          | 0.2156 | 0.2971 |
| 0.2053        | 95.0  | 57285 | 0.1804          | 0.1804 | 0.2756 |
| 0.218         | 96.0  | 57888 | 0.2050          | 0.2050 | 0.2826 |
| 0.21          | 97.0  | 58491 | 0.2026          | 0.2026 | 0.2888 |
| 0.2071        | 98.0  | 59094 | 0.2169          | 0.2169 | 0.2976 |
| 0.2039        | 99.0  | 59697 | 0.2475          | 0.2475 | 0.3016 |
| 0.2093        | 100.0 | 60300 | 0.1992          | 0.1992 | 0.2873 |


### Framework versions

- Transformers 4.26.0
- Pytorch 1.13.1
- Datasets 2.8.0
- Tokenizers 0.13.2