---
title: "Introduction"
weight: 5
---

Mining and construction of complex structures have been important tasks for many centuries. They share increased requirements for the tools used, particularly drilling rigs. Often, for extracting natural gas, oil, or building metro lines, it is necessary to create deep shafts in hard heterogeneous rock, where the drill head is subjected to strong pressure and temperature. For such work, the characteristics of drill head tips must match the environmental requirements as accurately as possible. For example, carbide tips must be hard enough to effectively break rock, but also sufficiently ductile to avoid cracking when hitting a random high-strength stone.
    
With the development of materials science, it has been possible to identify a group of hard tungsten carbide-cobalt (WC-Co) alloys that have high and flexible strength characteristics. For example, an alloy will be more ductile and less prone to grain chipping if more binder (cobalt) is added. Over many years, all characteristics of this alloy have been studied in sufficient detail; however, we cannot claim that all its properties have been studied.

There is a hypothesis that there are nonlinear dependencies between the physical characteristics of the alloy. There are many works studying WC-Co alloys with different component percentages; however, the topic of finding hidden relationships remains unexplored. The goal of this work is to develop a software complex for extracting important features from WC-Co microstructure images and finding dependencies between alloy characteristics.

The most promising methods for finding dependencies are machine learning and computer vision algorithms. Modern neural network models are capable of identifying complex dependencies between very different types of data. Computer vision has many algorithms that effectively extract geometry from images.

To achieve the goal, it is necessary to research and test a large set of feature extraction algorithms on alloy microstructure images. Then, the obtained features need to be correlated with alloy characteristics such as: hardness, impact toughness, ductility, shock adiabat coefficients, etc. It is also important to conduct alloy modeling in various CAD systems to compare computed parameters with experimental ones.

The expected result of the work is a software complex for processing SEM images of WC-Co alloy microstructures and analyzing hidden dependencies between physical characteristics. 

This work has clear practical value. If hidden dependencies indeed exist, the mathematical model of the WC-Co alloy will be supplemented with new data. As a consequence, engineers manufacturing and applying carbide tips will know more about the behavior of this alloy during sintering and operation under various external conditions. This will help reduce costs and increase the service life of drilling rigs. 

It is important to understand that the value of the work will not decrease if the sought hidden dependencies are not found. In the future, the developed software complex can be used for image analysis, other alloys, or other subject areas such as biology and astronomy.
