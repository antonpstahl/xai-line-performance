# Failure slices (holdout)

High-risk parts (score >= 0.182 or top 2%) clustered by
their SHAP vectors. `mean_shap` is the mean log-odds contribution within the slice.

## Slice 1

- parts: 2610, failures: 10, failure rate: 0.0038 (all high-risk parts: 0.0243)
- flagged by threshold: 4, precision: 0.0
- top features: L0_S7_F142 (+0.092), L0_S7_F138 (+0.062), L3_S33_F3865 (+0.054), L0_S0_F10 (+0.053), L0_S3_F100 (+0.053)
- top stations: L3_S30 (+0.34), L0_S0 (+0.197), L3_S29 (+0.162)

> Slice 1 contains 2610 parts, of which 10 failed, a failure rate of 0.0038, and 4 parts were flagged above the decision threshold with a precision of 0.0, meaning none of the flagged parts failed. The strongest average contributions to predicted risk came from the features L0_S7_F142 (mean_shap 0.092), L0_S7_F138 (0.062), and L3_S33_F3865 (0.054), with the stations L3_S30 (0.34), L0_S0 (0.197), and L3_S29 (0.162) contributing most at the station level. For reference, the failure rate across all high-risk parts in every slice together is 0.0243, which is not this slice's rate, and this slice's failure rate of 0.0038 is well below that reference.

Faithfulness check: passed

## Slice 2

- parts: 807, failures: 75, failure rate: 0.0929 (all high-risk parts: 0.0243)
- flagged by threshold: 256, precision: 0.0781
- top features: L3_S32_F3850 (+1.256), L3_S32_elapsed (+0.639), L3_S34_F3882 (+0.422), L3_S33_F3865 (+0.386), L3_S32_visited (+0.208)
- top stations: L3_S32 (+2.103), L3_S33 (+1.094), L3_S34 (+0.443)

> Slice 2 contains 807 parts, of which 75 failed, giving a failure rate of 0.0929, and 256 parts were flagged above the decision threshold with a precision of 0.0781. The highest mean SHAP contributions to failure log-odds came from L3_S32_F3850 (1.256), L3_S32_elapsed (0.639), and L3_S34_F3882 (0.422), while L3_S32 (2.103), L3_S33 (1.094), and L3_S34 (0.443) were the top stations by mean SHAP. For reference, the failure rate across all high-risk parts in every slice together is 0.0243, which is a different baseline from this slice's rate of 0.0929.

Faithfulness check: passed

## Slice 3

- parts: 539, failures: 1, failure rate: 0.0019 (all high-risk parts: 0.0243)
- flagged by threshold: 1, precision: 0.0
- top features: L0_S0_F2 (+0.738), L3_S29_F3412 (+0.283), L0_S7_F142 (+0.148), L0_S1_F24 (+0.134), L3_S29_F3436 (+0.085)
- top stations: L0_S0 (+0.693), L3_S29 (+0.385), L3_S30 (+0.262)

> Slice 3 contains 539 parts with 1 failure, a failure rate of 0.0019, and 1 part was flagged above the decision threshold, but that flagged part did not fail, so precision is 0.0. The largest average contributions to failure log-odds come from L0_S0_F2 (mean_shap 0.738) and L3_S29_F3412 (mean_shap 0.283), with station L0_S0 at 0.693, L3_S29 at 0.385, and L3_S30 at 0.262. The high-risk failure rate across all slices is 0.0243, which serves only as a reference, and with a single failure in this slice, these feature contributions reflect the model's scoring rather than a confirmed failure pattern.

Faithfulness check: passed

## Slice 4

- parts: 421, failures: 0, failure rate: 0.0 (all high-risk parts: 0.0243)
- flagged by threshold: 8, precision: 0.0
- top features: L2_S27_F3166 (+0.765), duration (+0.523), L3_S29_F3412 (+0.459), L2_S27_F3129 (+0.17), L0_S0_F10 (+0.113)
- top stations: L2_S27 (+1.049), part-level (+0.523), L3_S29 (+0.386)

> Slice 4 contains 421 parts with 0 failures (a 0.0 failure rate), and the model flagged 8 of them, none of which failed (precision 0.0). The strongest upward risk contributions came from L2_S27_F3166 (mean_shap 0.765), duration (0.523), and L3_S29_F3412 (0.459), with L2_S27 the highest-contributing station at 1.049. For reference, the high-risk failure rate across all slices together is 2.43%, so this slice's zero failures do not reflect a rate on that scale and the flagged parts here did not show failures in the data provided.

Faithfulness check: passed

## Slice 5

- parts: 358, failures: 29, failure rate: 0.081 (all high-risk parts: 0.0243)
- flagged by threshold: 20, precision: 0.0
- top features: L3_S32_F3850 (+0.764), L3_S32_elapsed (+0.662), L3_S33_F3865 (+0.389), L3_S33_F3855 (+0.361), L3_S32_visited (+0.211)
- top stations: L3_S32 (+1.637), L3_S33 (+1.252), L3_S38 (+0.052)

> Slice 5 contains 358 parts with 29 failures, a failure rate of 8.1%. Of the 20 parts flagged above the decision threshold, none failed, giving a precision of 0%.

The largest positive contributions come from L3_S32_F3850 (mean_shap 0.764) and L3_S32_elapsed (mean_shap 0.662), and at the station level L3_S32 (1.637) and L3_S33 (1.252) contribute most, while L3_S38 contributes little (0.052).

For reference, the failure rate across all high-risk parts combined is 2.43%, so this slice's 8.1% is higher, but none of the flagged parts here actually failed, so the model's risk signal for this slice is not yet confirmed by outcomes.

Faithfulness check: passed
