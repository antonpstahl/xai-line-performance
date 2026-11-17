# Failure slices (holdout)

High-risk parts (score >= 0.330 or top 2%) clustered by
their SHAP vectors. `mean_shap` is the mean log-odds contribution within the slice.

## Slice 1

- parts: 165, failures: 81, failure rate: 0.4909 (all high-risk parts: 0.2565)
- flagged by threshold: 163, precision: 0.4969
- top features: L3_S32_visited (+0.645), L3_S32_F3850 (+0.621), L3_S32_time (+0.62), L3_S33_F3865 (+0.329), L3_S33_F3857 (+0.299)
- top stations: L3_S32 (+1.887), L3_S33 (+1.225), L3_S34 (+0.345)

## Slice 2

- parts: 117, failures: 7, failure rate: 0.0598 (all high-risk parts: 0.2565)
- flagged by threshold: 4, precision: 0.0
- top features: L0_S11_time (+0.205), L0_S8_time (+0.162), L3_S30_F3704 (+0.114), L0_S0_time (+0.107), L3_S30_F3644 (+0.089)
- top stations: L3_S30 (+0.696), L0_S11 (+0.47), L0_S0 (+0.269)

## Slice 3

- parts: 88, failures: 7, failure rate: 0.0795 (all high-risk parts: 0.2565)
- flagged by threshold: 1, precision: 0.0
- top features: L0_S15_time (+0.134), L0_S18_time (+0.126), L0_S12_time (+0.093), L3_S30_time (+0.055), L0_S12_F330 (+0.05)
- top stations: L0_S12 (+0.267), L3_S29 (+0.228), L3_S30 (+0.219)

## Slice 4

- parts: 74, failures: 4, failure rate: 0.0541 (all high-risk parts: 0.2565)
- flagged by threshold: 0, precision: None
- top features: L0_S8_time (+0.093), L0_S4_time (+0.089), L3_S30_time (+0.074), L3_S33_F3865 (+0.062), L3_S30_F3624 (+0.062)
- top stations: L3_S30 (+0.625), L3_S29 (+0.259), L0_S0 (+0.22)

## Slice 5

- parts: 55, failures: 29, failure rate: 0.5273 (all high-risk parts: 0.2565)
- flagged by threshold: 38, precision: 0.6053
- top features: L3_S32_visited (+0.578), L3_S32_F3850 (+0.533), L3_S32_time (+0.525), L3_S33_F3865 (+0.35), L3_S33_F3857 (+0.272)
- top stations: L3_S32 (+1.636), L3_S33 (+1.204), L3_S30 (+0.154)
