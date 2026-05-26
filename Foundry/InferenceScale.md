# How to Scale Inference of model with Microsoft Foundry

## Overview

- Idea here is to scale the model deployed with different patterns for production
- Every PaaS model inferencing has some limitation in TPM, or request and others
- It's also helpful to create High availability and fault tolerance in the deployment.
- Looking at deploying with multiple regions
- Avoid limitation based on subscription limits if needed.
- Avoid huge Latency in a specific region

## Patterns

### Microsoft Foundry Horizontal Scale for Model Inferencing – Same Subscription with Multiple region

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundryinfscale-1.jpg 'fine tuning model')

- Above model allows us to scale the model across multiple regions within the same subscription.
- How the scale is implemented depends on the specific requirements and constraints of the deployment.
- For simple we can use Round Robin, or circuit breaker based on Latency.

### Microsoft Foundry Horizontal Scale for Model Inferencing – Multiple Subscription with Multiple region

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundryinfscale-2.jpg 'fine tuning model')

- This is the best model , which provides the highest level of scalability and fault tolerance.
- With multiple subscriptions, we can distribute the load across different regions and avoid any single point of failure.
- We also use multiple regions to ensure high availability and fault tolerance.
- By using multiple subscription we can also avoid subscriptions limits.

### Microsoft Foundry Horizontal Scale for Model Inferencing – Different Subscription with Same region

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundryinfscale-3.jpg 'fine tuning model')

- This pattern is advisable for data sovernity compliance.
- If only one region is to be used, then this pattern ensures that the data remains within the boundaries of that region.
- This pattern also helps in reducing the latency by ensuring that the model is deployed closer to the users.

## Conclusion

- We saw few patterns on how we can scale model inferencing across multiple regions and subscriptions.
- Provide higher availbility and latency for production ready application
- Using the PayGo model, we can pay for the resources we use without any upfront commitment.
- Also to avoid individual region limitations with usage and traffic.
- Ability to scale the model based on the demand and usage patterns.