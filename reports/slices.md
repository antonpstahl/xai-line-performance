# Failure slices (holdout)

High-risk parts (score >= 0.182 or top 2%) clustered by
their SHAP vectors. `mean_shap` is the mean log-odds contribution within the slice.

## Slice 1

- parts: 2610, failures: 10, failure rate: 0.0038 (all high-risk parts: 0.0243)
- flagged by threshold: 4, precision: 0.0
- top features: L0_S7_F142 (+0.092), L0_S7_F138 (+0.062), L3_S33_F3865 (+0.054), L0_S0_F10 (+0.053), L0_S3_F100 (+0.053)
- top stations: L3_S30 (+0.34), L0_S0 (+0.197), L3_S29 (+0.162)

## Slice 2

- parts: 807, failures: 75, failure rate: 0.0929 (all high-risk parts: 0.0243)
- flagged by threshold: 256, precision: 0.0781
- top features: L3_S32_F3850 (+1.256), L3_S32_elapsed (+0.639), L3_S34_F3882 (+0.422), L3_S33_F3865 (+0.386), L3_S32_visited (+0.208)
- top stations: L3_S32 (+2.103), L3_S33 (+1.094), L3_S34 (+0.443)

## Slice 3

- parts: 539, failures: 1, failure rate: 0.0019 (all high-risk parts: 0.0243)
- flagged by threshold: 1, precision: 0.0
- top features: L0_S0_F2 (+0.738), L3_S29_F3412 (+0.283), L0_S7_F142 (+0.148), L0_S1_F24 (+0.134), L3_S29_F3436 (+0.085)
- top stations: L0_S0 (+0.693), L3_S29 (+0.385), L3_S30 (+0.262)

## Slice 4

- parts: 421, failures: 0, failure rate: 0.0 (all high-risk parts: 0.0243)
- flagged by threshold: 8, precision: 0.0
- top features: L2_S27_F3166 (+0.765), duration (+0.523), L3_S29_F3412 (+0.459), L2_S27_F3129 (+0.17), L0_S0_F10 (+0.113)
- top stations: L2_S27 (+1.049), part-level (+0.523), L3_S29 (+0.386)

## Slice 5

- parts: 358, failures: 29, failure rate: 0.081 (all high-risk parts: 0.0243)
- flagged by threshold: 20, precision: 0.0
- top features: L3_S32_F3850 (+0.764), L3_S32_elapsed (+0.662), L3_S33_F3865 (+0.389), L3_S33_F3855 (+0.361), L3_S32_visited (+0.211)
- top stations: L3_S32 (+1.637), L3_S33 (+1.252), L3_S38 (+0.052)
