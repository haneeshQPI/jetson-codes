# ONNX Model Conversion Codebase Visual Map

## System Architecture Overview

```mermaid
graph TB
    subgraph "CI/CD Pipeline"
        GITLAB[GitLab CI/CD]
        ACR[Azure Container Registry]
        AKS[Azure Kubernetes Service]
    end

    subgraph "Core Application"
        MAIN[app/main.py]
        API[app/api/v1.py]
        CONFIG[app/core/config.py]
    end

    subgraph "Edge Deployment Core"
        PROCESS[edge_deploy/process.py]
        UTILS[edge_deploy/utils.py]
        CONFIG_EDGE[edge_deploy/config.py]
        RUN[edge_deploy/run.py]
    end

    subgraph "Database Layer"
        MONGODB[(MongoDB)]
        REDIS[(Redis)]
    end

    subgraph "Docker Build Process"
        DOCKER_BUILD[Docker Image Build]
        MODEL_DECRYPT[Model Decryption]
        PLATFORM_DETECT[Platform Detection]
        IMAGE_PUSH[Image Push]
    end

    subgraph "ONNX Conversion Workflow"
        FLYTE_TASK[tasks/modelconversion.py]
        MM_DEPLOY[MMDeploy Engine]
        ONNX_OUTPUT[ONNX Models]
    end

    subgraph "Storage & Artifacts"
        LAKEFS_S3[(LakeFS/S3 Storage)]
        MODEL_ARTIFACTS[Model Artifacts]
        EDGE_ARTIFACTS[Edge Deployment Artifacts]
    end

    %% CI/CD Flow
    GITLAB --> ACR
    ACR --> AKS

    %% Application Flow
    MAIN --> API
    MAIN --> CONFIG
    API --> PROCESS

    %% Edge Deployment Flow
    PROCESS --> UTILS
    PROCESS --> CONFIG_EDGE
    PROCESS --> MONGODB
    PROCESS --> REDIS
    UTILS --> RUN
    RUN --> DOCKER_BUILD

    %% Docker Build Flow
    DOCKER_BUILD --> MODEL_DECRYPT
    DOCKER_BUILD --> PLATFORM_DETECT
    DOCKER_BUILD --> IMAGE_PUSH

    %% ONNX Conversion Flow
    FLYTE_TASK --> MM_DEPLOY
    MM_DEPLOY --> ONNX_OUTPUT
    ONNX_OUTPUT --> EDGE_ARTIFACTS

    %% Storage Flow
    MODEL_ARTIFACTS --> LAKEFS_S3
    EDGE_ARTIFACTS --> LAKEFS_S3
    REDIS --> MONGODB

    %% Styling
    classDef ciClass fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef appClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef edgeClass fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef dbClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef dockerClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef workflowClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef storageClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class GITLAB,ACR,AKS ciClass
    class MAIN,API,CONFIG appClass
    class PROCESS,UTILS,CONFIG_EDGE,RUN edgeClass
    class MONGODB,REDIS dbClass
    class DOCKER_BUILD,MODEL_DECRYPT,PLATFORM_DETECT,IMAGE_PUSH dockerClass
    class FLYTE_TASK,MM_DEPLOY,ONNX_OUTPUT workflowClass
    class LAKEFS_S3,MODEL_ARTIFACTS,EDGE_ARTIFACTS storageClass
```

## Detailed Component Breakdown

### 1. CI/CD Pipeline Layer
- **GitLab CI/CD**: Manages the build and deployment pipeline
- **Azure Container Registry (ACR)**: Stores Docker images
- **Azure Kubernetes Service (AKS)**: Orchestrates container deployments

### 2. Core Application Layer
- **`app/main.py`**: Main application entry point
- **`app/api/v1.py`**: API endpoint definitions
- **`app/core/config.py`**: Application configuration management

### 3. Edge Deployment Core Layer
- **`edge_deploy/process.py`**: Central processing logic for edge deployments
- **`edge_deploy/utils.py`**: Utility functions for deployment operations
- **`edge_deploy/config.py`**: Edge deployment specific configuration
- **`edge_deploy/run.py`**: Execution engine for deployment tasks

### 4. Database Layer
- **MongoDB**: Primary document database for storing deployment metadata
- **Redis**: In-memory cache and message broker for deployment operations

### 5. Docker Build Process Layer
- **Docker Image Build**: Main build orchestration
- **Model Decryption**: Handles encrypted model files
- **Platform Detection**: Identifies target deployment platforms
- **Image Push**: Pushes built images to registry

### 6. ONNX Conversion Workflow Layer
- **`tasks/modelconversion.py`**: Flyte task for model conversion
- **MMDeploy Engine**: Core conversion engine
- **ONNX Models**: Output converted models

### 7. Storage & Artifacts Layer
- **LakeFS/S3 Storage**: Object storage for models and artifacts
- **Model Artifacts**: Input PyTorch models and configurations
- **Edge Deployment Artifacts**: Output ONNX models and deployment files

## Data Flow

1. **Model Input**: PyTorch models are stored in LakeFS/S3 under `{branch}/model/`
2. **Conversion Process**: The Flyte task downloads models and converts them using MMDeploy
3. **Output Generation**: ONNX models are generated and stored locally
4. **Artifact Upload**: Converted models are uploaded to `{branch}/edge/` in LakeFS/S3
5. **Deployment**: Edge deployment core processes the artifacts for deployment

## Key Technologies

- **Flyte**: Workflow orchestration
- **MMDeploy**: Model deployment framework
- **Docker**: Containerization
- **LakeFS/S3**: Object storage
- **MongoDB**: Document database
- **Redis**: Caching and messaging
- **GitLab CI/CD**: Continuous integration/deployment

## Security Considerations

- Credentials are managed through environment variables
- No hardcoded secrets in the codebase
- Secure communication with storage services
- Container-based isolation for model processing
