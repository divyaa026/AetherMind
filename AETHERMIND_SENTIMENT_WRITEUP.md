# AetherMind: Real-Time Sentiment Analysis Platform

**Tech stack:** Python, Transformers, DistilBERT, Flask, Docker, TorchServe

## 1. Problem Statement

Businesses that rely on user feedback, support tickets, or social/streaming text data need to understand sentiment as it arrives, not hours or days later in a batch report. Most sentiment analysis setups either run as a single offline script (too slow and brittle for live traffic) or as a single monolithic service (hard to scale, and a single point of failure when traffic spikes). The goal of this project was to build a sentiment classification system that could ingest text continuously, classify it accurately, and return results fast enough to support real-time decision-making — while remaining reliable and scalable enough for bursty, unpredictable traffic.

## 2. What I Did

- **Fine-tuned a transformer model for the task.** Took DistilBERT, a lightweight distilled version of BERT, and fine-tuned it on a custom, hand-labelled dataset of 10,000+ text records for sentiment classification — balancing model accuracy against inference speed, since DistilBERT retains most of BERT's language understanding at roughly 60% of the size and latency cost.
- **Designed a containerized microservices architecture** instead of a single monolithic app, splitting the system into independently deployable services: an API gateway to handle incoming requests and routing, and dedicated inference services for model serving.
- **Served the model with TorchServe** for streaming data ingestion and inference, and load-balanced traffic across multiple inference endpoints so the system could absorb bursts of traffic without degrading response times.
- **Containerized every service with Docker**, so each component (gateway, ingestion, inference) could be built, deployed, and scaled independently rather than as one large deployable unit.
- **Built a Flask-based API layer** to expose the sentiment classification functionality as a clean, consumable service.
- **Automated model maintenance** by setting up a scheduled retraining loop with drift detection, orchestrated via GitHub Actions, so the model could be monitored and refreshed over time as incoming data patterns shifted, rather than silently degrading in production.

## 3. Result and Impact

- Achieved **85% classification accuracy** on real-time sentiment predictions, validated against the 10,000+ record labelled dataset.
- Delivered **sub-100ms inference latency**, making the system suitable for real-time, user-facing use cases rather than just offline batch analysis.
- The microservices design meant the system could **handle burst traffic gracefully** and scale individual components (e.g., inference) independently of the rest of the stack, instead of scaling the whole application at once.
- The automated drift-detection and retraining loop reduced the operational burden of keeping the model accurate over time, moving model maintenance from a manual, reactive process to a scheduled, automated one.
- Overall, the project demonstrated an end-to-end MLOps workflow — from fine-tuning a transformer model, through containerized deployment, to automated monitoring and retraining — going beyond a notebook-only ML project into a production-style deployment.
