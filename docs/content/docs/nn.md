---
title: "Machine Learning"
weight: 20
---

To use images, you need to create an autoencoder model that will extract a feature vector from images. Then this vector together with computed characteristics will be sent to the input of another neural network, which will search for dependencies between input values and physical characteristics of alloys.

## Architecture:

![Pipeline](https://pobedit.s3.us-east-2.amazonaws.com/docs_images/pipeline.jpg)

[AE weights (tf)](https://pobedit.s3.us-east-2.amazonaws.com/ml_weights/u2net_2021-11-19.h5)

[VQ-VAE-2 weights (pt)](https://pobedit.s3.us-east-2.amazonaws.com/ml_weights/vqvae_002_train_0.00976_test_0.00967_1024.pt)
