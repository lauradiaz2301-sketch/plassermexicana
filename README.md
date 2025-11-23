# README

## Predictive Maintenance for Railway Systems
### Problem Statement
Train system operators face a significant financial challenge: current logistics lack predictive maintenance capabilities. Millions are spent deploying heavy machinery without prior knowledge of track conditions, leading to inefficient resource allocation and unnecessary costs.

### Our Solution
We developed a system that combines:

- Ground data from Plasser machinery

- Satellite observations

- Environmental data

- Predictive modeling

To answer the critical question: When and where is maintenance intervention most valuable?

### How It Works
Our approach uses a two-stage machine learning pipeline:

**1. Ground-Level Classification** 
Input: Plasser's ground measurements

Algorithm: Random Forest Classifier

Output: Binary probability classification for maintenance needs

Purpose: Identifies patterns requiring immediate attention

![WhatsApp Image 2025-11-23 at 10 23 47](https://github.com/user-attachments/assets/e3beaa52-e1ed-464a-bc13-e1b3b8fefc17)


[**2. Orbital-Level Regression**](https://9000-firebase-studio-1763893024021.cluster-ocv3ypmyqfbqysslgd7zlhmxek.cloudworkstations.dev/dashboard)

Input: Satellite and environmental data

Algorithm: Random Forest Regressor

Output: Probabilistic estimation of maintenance percentage

Purpose: Pinpoints areas requiring closer inspection

![WhatsApp Image 2025-11-23 at 12 21 26](https://github.com/user-attachments/assets/605afb03-9ad7-4988-bbfa-71fba93aef7d)

### Dataset
The synthetic dataset used in this project was generated based on parameters from the TAMP thesis research:

[Track Geometry and Material Properties (TAMP) - TU Wien](https://www.tuwien.at/en/cee/geotechnik/igb/research/completed-research-projects/tamp#:~:text=The%20investigations%20carried%20out%20within,%C2%A9%20Plasser%20&%20Theurer)

### Key Benefits
- Reduced unnecessary deployments
- Fewer maintenance hours
- Increased track availability
- Data-driven decision making

### System Integration
Our system combines:
**Orbital intelligence:** Potential problem areas identified from space
**Maintenance history:** Past and current intervention records

## The result is simple: More efficient maintenance operations and significant cost savings.


