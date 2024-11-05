```mermaid
---
title: Pipeline
---
classDiagram
    Trainer <|-- BayesModelCls 
    Trainer <|-- OptimizationParams
    NetDistribution <|--Trainer
    Ensemble <|--NetDistribution
    CompressedNet <|--NetDistribution
    CompressedNet <|-- Compressor

    class NetDistribution{

    }
    class BayesModelCls{

    }
    class OptimizationParams{
    }
    class Trainer{
    }
    class Ensemble{
    }
    class Compressor{
    }
    class CompressedNet{
    }
```