# AI Email Operations Environment

## Overview
This project simulates a real-world enterprise email workflow system where AI agents must classify, prioritize, and respond to emails.

## Motivation
Modern organizations face email overload. Automating triage and response improves efficiency and reduces human workload.

## Tasks
- Easy: Email classification
- Medium: Classification + response generation
- Hard: Full workflow automation (classification, prioritization, response)

## Dataset
Derived from Enron Email Dataset with preprocessing, labeling, and augmentation using real-world scenarios.

## Reward Design
The reward function evaluates:
- Classification accuracy
- Priority handling
- Response quality
- Behavioral consistency

## Key Features
- Real-world dataset
- Multi-step decision making
- Partial reward shaping
- Edge case handling

## Use Cases
- Training AI assistants
- Evaluating LLM agents
- Enterprise workflow automation research

## Baseline Performance
Reproducible via inference.py