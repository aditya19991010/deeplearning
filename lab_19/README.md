# Tutorial Generataive adversial network

---

## Tutorial by Pytorch on DCGAN
[Link](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)


## Projects on GAN
[Link](https://github.com/nashory/gans-awesome-applications)

## Steps


```mermaid
FLOWCHART

A[Start] --> B[Import Libraries]
B --> C[Load & Preprocess Dataset]
C --> D[Define Hyperparameters]

D --> E[Build Generator Model]
E --> F[Build Discriminator Model]

F --> G[Define Loss Functions]
G --> H[Define Optimizers for G and D]

H --> I[Combine Generator + Discriminator (Adversarial Model)]
I --> J[Start Training Loop]

J --> K[For each epoch:]
K --> L[1. Sample real data batch]
L --> M[2. Generate fake data batch using Generator]
M --> N[3. Train Discriminator on real and fake data]
N --> O[4. Train Generator via combined model (keeping D frozen)]

O --> P[5. Calculate & Log Loss/Accuracy]
P --> Q[6. Save Generated Samples for Monitoring]
Q --> R[7. Update Model Weights]

R --> S{Converged?}
S -->|No| K
S -->|Yes| T[Save Final Generator Model]
T --> U[Generate & Visualize New Data]

U --> V[End]
