..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard, Damien Guenot, Charlie Vanaret

Glossary
--------

.. glossary::
    :sorted:

    MDA
        Multi-Disciplinary Analysis

    MDO
        Multi-Disciplinary Design Optimization

    MDF
        Multi-Disciplinary Feasible

    IDF
        Individual Discipline Feasible

    BLISS
        Bi-Level Integrated System Synthesis,

    BLISS98
        Bi-Level Integrated System Synthesis, original BLISS variant from 1998 paper using coupled derivatives

    BLISS2000
        Bi-Level Integrated System Synthesis, BLISS variant from 2000 paper, using surrogate models

    CO
        Collaborative Optimization

    ANN
        Artificial Neural Network

    CSSO
        Concurrent Subspaces Optimization

    MDOIS
        MDO of Independent Subspaces

    AAO
        All At Once

    ASO
        Asymmetric Subspace Optimization

    SAND
        Simultaneous Analysis and Design

    ATC
        Analytical Target Cascading

    WORMS
        Workflow Management System

    eWORMS
        eclipse Workflow Management System

    CFD
        Computational Fluid Dynamics

    CSM
        Computational Structure Mechanics

    EA
        Evolutionary Algorithm

    SBO
        Surrogate Based Optimization

    XDSM
        eXtended Design Structure Matrix

    LP
        Linear Programming

    LHS
        Latin Hypercube Sampling

    SOM
        Self Organizing Map

    API
        Application Program Interface

    COTS
        Commercial Off The Shelf

    SRS
        Software Requirements Specification

    HPC
        High Performance Computing

    UML
        Unified Modeling Language

    IRT
        Institut de Recherche Technologique Saint Exupéry

    DFO
        Derivative-Free Optimization

    ANR
        Agence Nationale de la Recherche

    KKT
        Karun Kuhn Tucker

    LOO
        Leave One Out

    RBF
        Radial Basis Function

    |g|
        Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization

    JSON
        JavaScript Object Notation

    XML
        Extensible Markup Language

    SSBJ
        SuperSonic Business Jet

    MMA
        Method of Moving Asymptotes

    SLSQP
        Sequential Least Squares Quadratic Programming

    FPO
        Future Projects Office

    HTML
        Hypertext Markup Language

    SWIG
        Simplified Wrapper and Interface Generator

    PIDO
        Process Integration and Design Optimization

    SLM
        Simulation Life Cycle Management

    DOE
        Design Of Experiments

    HDF
        Hierarchical Data Format to save and structure files with huge data

    grammar
        A set of rules to be respected by a data set. Typically used to describe the inputs and outputs of a discipline

    JSON schema
        A JSON description of JSON data, similar to XML schemas

    process
        A series of executions and data exchanges, ie the workflow and the data flow.

    workflow engine
        A program used to design, run and analyze processes

    processes data
        The disciplines inputs and generated outputs during the execution of a process

    serialization
        Process of writing objects or data structures to disk or more generally, to formats that can be stored

    work flow
        The execution sequence of the disciplines in a process

    data flow
        The sequence of data creation by, and exchanges between disciplines in a process

    MDO integrator
        A user class that uses disciplines and MDO formulations to create, test and maintain MDO scenarios

    disciplinary expert
        A user class of the MDO platform that wraps disciplinary capabilities into disciplines and creates, tests or maintains disciplinary processes

    MDO user
        A user class of the MDO platform that executes an MDO scenario to produce results

    MDO formulations designer
        A user class of |g| that creates, implements, tests or maintains MDO formulations

    algorithm integrator
        A user class of the formulation engine that integrates mathematical algorithms such as an optimization algorithm, :term:`DOE` method or surrogate model

    discipline
        One program, or an arbitrary set of simulation software, that can be viewed as a mathematical multivalued function, taking inputs and producing outputs through its execution

    chain
        A process that executes a set of disciplines in a sequential way, where outputs of the previously executed disciplines are passed as inputs of the next ones

    simulation software
        A program that simulates a part of the physics of a system, or contributes to the overall simulation of the system, such as a mesher

    MDO formulations engine
        A program that enables the implementation of MDO formulations

    wrapper
        Here discipline wrappers. Standardized interface defining inputs, outputs and execution of a given simulation software.

    library wrapper
        A code that translates the existing API of a program, or a library, into a compatible one

    interface
        A set of functions and data from a software exposed to other software

    design problem
        An engineering problem such that a shape has to be changed to match or improve criteria under constraints

    MDO formulation
        The mathematical strategy used to define the optimization problem(s) to be solved

    bi-level
        A type of :term:`MDO formulation` which formulates multiple optimization problems

    Monolithic
        A type of :term:`MDO formulation` which formulates a single optimization problem

    disciplinary optimization
        The most basic MDO formulation restricted to a single set of design variables and only suited for a :term:`weakly coupled problem`

    weakly coupled problem
        A multidisciplinary problem where the coupling variables can be computed by a single execution :term:`chain`

    MDO architecture
        the software architecture that enables the programming and resolution of MDO design problems

    scenario
        The translation of a design problem into an executable. When executed, a scenario generates a :term:`process`

    Design Of Experiments
        A sampling of a design space, or a generic method that produces samplings of design spaces

    design space
        The mathematical set containing the design variables of an optimization problem

    design variables
        The unknowns of the optimization problem

    system design variables
        The design variables that are shared by more than one discipline, at the system level optimization problem in a bi level MDO scenario

    shared design variables
        The design variables that are shared by more than one discipline

    coupling variables
        In an MDO scenario, variables that are both used as inputs of a discipline and outputs of another one, or the same discipline

    local design variables
        In an MDO scenario, a subset of the design variables that are inputs of only one discipline

    disciplinary design variables
        In an MDO scenario, a subset of the design variables that are inputs of only one discipline

    operating condition
        An input parameter of a simulation program that defines a physical parameter in which the system operates. Typically the speed of a vehicle, or its altitude

    MDO platform
        A set of programs integrated in a common framework, enabling the resolution of multidisciplinary engineering problems using numerical simulation and optimization

    generic process
        A process that can be applied to any discipline or set of disciplines, such as :ref:`mda` methods that solves the coupling variables of a set of disciplines

    optimization problem
        A mathematical problem consisting in finding a set of variables which minimizes or maximizes a mathematical function (possibly a set-valued map), while satisfying constraints on these variables or on artrary functions

    optimization history
        The database of values of the objective function, constraints and design variables obtained during an optimization

    optimization algorithm
        An algorithm capable of solving optimization problems

    DOE algorithm
        An algorithm that generates samples of the design space

    trade-off
        A study that aims at comparing different options in terms of design parameters, and analyse their impacts

    driver
        A :term:`optimization algorithm` or :term:`DOE`

    objective function
        The function to be minimized or maximized in an optimization problem

    constraint
        A function of the design variables that must be kept either null or negative in an optimization problem

    constraints
        All the functions of the design variables that must be kept either null or negative in an optimization problem

    surrogate model
        A mathematical model of another model. Typically used to substitute an expensive simulation-based model by an approximation whose cost of evaluation is lower, at the price of an initial sampling of the original model

    workflow-driven
        Characteristic of a workflow engine, for which processes are described through the work flow, and in which the data flow is deduced accordingly

    data-driven
        Characteristic of a workflow engine, for which processes are described through the data flow, and in which the work flow is deduced accordingly

    modular architecture
        A software architecture based on separated components, with a relative independence between them

    Model Center
        COTS workflow engine developed by Phoenix Integration

    Scilab
        An open-source alternative to Matlab

    LSF
        COTS jobs scheduler for HPC clusters

    NLopt
        Non-Linear Optimization package, an open source library of optimization algorithms from MIT, http://ab-initio.mit.edu/nlopt

    SciPy
        Open Source Library of Scientific Tools, containing a library of optimization algorithms, https://www.scipy.org

    Secure Shell
        (SSH) an encrypted network protocol for accessing remote computers

    sequence diagram
        A :term:`UML` diagram that shows objects execution sequence, such as the function calls and their arguments, and the execution order

    job scheduler
        A program used to distribute tasks and allocate resources for tasks on HPC clusters

    complex step
        A numerical method to approximate the derivative of a function, similarly to finite differences, but using complex perturbations

    finite differences
        A numerical method to approximate the derivative of a function by small perturbations of the inputs

    design document
        A document that describes the design of a program, responding to a Software Requirements Specification

    OpenDACE
        Open Design and Analysis of Computer Experiments: a program developed by Airbus to standardize interfaces of optimization algorithms, :term:`DOE` methods and surrogate models as well as the related problems to be solved

    gradient-based optimization
        A class of optimization algorithms that use the total derivatives of the objective function and constraints

    gradient
        Total derivative of a function with respect to its variables

    jacobian
        The matrix of first order partial derivatives of outputs with respect to inputs

    Hessian
        The matrix of second order partial derivatives of one output with respect to inputs

    optimum
        Solution of an :term:`optimization problem`: the :term:`design variables` values at the minimum of the function, subject to the constraints

    Object Oriented Programming
        A programming paradigm based on objects, which are data structures as well as a structure for methods

    run time
        In computer science, run time, runtime or execution time is the time during which a program is running (executing), in contrast to other program life cycle phases such as compile time, link time and load time

    pip
        The PyPA recommended tool for installing Python packages. https://pypi.python.org/pypi/pip

    anaconda
        A python tool to create virtual environment and easily install precompiled packages, https://www.anaconda.com/distribution

    fixed-point
        A family of numerical resolution methods based on an iterative sequence of execution of the type : A() -> B() -> C() -> ... -> A() -> B() -> C() until convergence

    root finding
        A family of numerical resolution methods that solves multivariate problems of the type R(x)=0

    Newton method
        A :term:`root finding` method that uses successive linear approximations of the function of interest
