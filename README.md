# Email Triage OpenEnv

A real-world multi-step email triage environment for evaluating agent decision-making.

## Why this environment
Email triage is a real human workflow involving classification, prioritization, routing, disposition, escalation, and response drafting.

## Tasks
- easy: category + priority
- medium: category + priority + route + disposition
- hard: full triage including escalation and response drafting

## Action space
The agent must emit one structured action per stage:
- set_category
- set_priority
- set_route
- set_disposition
- set_escalation
- draft_response
- submit
