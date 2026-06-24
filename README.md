# Deep Learning Applications Labs

This repository contains the laboratory projects developed for the **Deep Learning Applications** course.

The repository is divided into three independent labs:

- **Homework01: From Pixels to Semantics** 
- **Homework02: The Transformative Transformer** 
- **Homework03: Getting Up to Speed with Deep Reinforcement Learning** 

## Dependency Management

Each lab is managed independently using **uv**.

Every project includes its own dependency configuration and virtual environment (`.venv`).

Install the dependencies of a specific lab with:

```bash
uv sync
```

## Repository Structure

The repository is organized as a collection of projects, one for each laboratory assignment.

```text
DLA_Homeworks/
│
├── Homework01/
│   ├── README.md
│   ├── results/
│   ├── DLA-Lab1.ipynb
│   ├── utils.py
│   ├── configResNet.yaml
│   └── FasterRCNN_config.yaml
│
├── Homework02/
│   ├── README.md
│   ├── DLA-Lab2.ipynb
│   ├── precompute_embeddings.py
│   └── app.py
│
├── Homework03/
│   ├── README.md
│   ├── Excercises_1_2.py
│   ├── agent.py
│   ├── car_race.py
│   ├── ppo.py
│   └── testAgent.py
│
└── README.md
```
## Additional Information

Each lab contains its own `README.md` file with detailed information about:

- Project objectives;
- Installation instructions;
- Usage examples;
- Implementation details.
- Information about the use of AI