# ONNX Model Conversion Workflow Diagram

## Core Workflow Process

```mermaid
flowchart TD
    A[PyTorch Models in LakeFS/S3<br/>{branch}/model/] --> B[Flyte Task Execution<br/>tasks/modelconversion.py]
    B --> C[Download Models<br/>boto3 S3 client]
    C --> D[Local Storage<br/>./download/]
    D --> E[MMDeploy Conversion<br/>ONNX Export]
    E --> F[Output Generation<br/>./mmdeploy_models/mmdet/ort/]
    F --> G[Upload to LakeFS/S3<br/>{branch}/edge/]
    
    subgraph "Input Layer"
        A
    end
    
    subgraph "Processing Layer"
        B
        C
        D
        E
        F
    end
    
    subgraph "Output Layer"
        G
    end
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
    style G fill:#f1f8e9
```

## Detailed Process Flow

```mermaid
sequenceDiagram
    participant User
    participant Flyte
    participant S3Client
    participant LocalFS
    participant MMDeploy
    participant LakeFS

    Note over User,LakeFS: ONNX Model Conversion Workflow
    
    User->>Flyte: Execute onnx_conversion task
    Flyte->>S3Client: Initialize boto3 client
    
    Note over S3Client: Step 1: Download Models
    S3Client->>LakeFS: List objects in {branch}/model/
    LakeFS->>S3Client: Return file list
    S3Client->>LocalFS: Download config.py and *.pth files
    LocalFS->>S3Client: Confirm download
    
    Note over MMDeploy: Step 2: Convert Models
    Flyte->>MMDeploy: Execute MMDeploy conversion
    MMDeploy->>LocalFS: Read model files
    LocalFS->>MMDeploy: Provide model data
    MMDeploy->>LocalFS: Generate ONNX files
    LocalFS->>Flyte: Confirm conversion
    
    Note over S3Client: Step 3: Upload Results
    Flyte->>S3Client: Upload converted artifacts
    S3Client->>LakeFS: Upload to {branch}/edge/
    LakeFS->>S3Client: Confirm upload
    S3Client->>Flyte: Return success status
    Flyte->>User: Task completion
```

## System Architecture

```mermaid
graph LR
    subgraph "Source Storage"
        LAKEFS[(LakeFS/S3<br/>Model Repository)]
    end
    
    subgraph "Processing Engine"
        FLYTE[Flyte Task<br/>modelconversion.py]
        MM_DEPLOY[MMDeploy<br/>Conversion Engine]
        DOCKER[Docker Container<br/>Ubuntu + MMDeploy + Flyte]
    end
    
    subgraph "Target Storage"
        EDGE_STORAGE[(LakeFS/S3<br/>Edge Repository)]
    end
    
    LAKEFS -->|Download| FLYTE
    FLYTE -->|Convert| MM_DEPLOY
    MM_DEPLOY -->|Generate| FLYTE
    FLYTE -->|Upload| EDGE_STORAGE
    
    DOCKER -.->|Hosts| FLYTE
    DOCKER -.->|Includes| MM_DEPLOY
    
    style LAKEFS fill:#e1f5fe
    style FLYTE fill:#f3e5f5
    style MM_DEPLOY fill:#e8f5e8
    style DOCKER fill:#fff3e0
    style EDGE_STORAGE fill:#f1f8e9
```

## Key Components & Dependencies

### Core Files
- **`tasks/modelconversion.py`**: Main Flyte task implementation
- **`Dockerfile`**: Container environment setup
- **`requirements.txt`**: Python dependencies
- **`.gitlab-ci.yml`**: CI/CD pipeline configuration

### External Dependencies
- **MMDeploy**: Model conversion framework
- **Flyte**: Workflow orchestration
- **boto3**: S3/LakeFS client
- **LakeFS/S3**: Object storage

### Data Flow
1. **Input**: PyTorch models from `{branch}/model/` in LakeFS
2. **Processing**: Local conversion using MMDeploy
3. **Output**: ONNX models uploaded to `{branch}/edge/` in LakeFS

### Key Features
- **Automated**: Single Flyte task handles entire workflow
- **Scalable**: Containerized execution
- **Integrated**: Seamless LakeFS/S3 integration
- **CI/CD Ready**: GitLab CI/CD pipeline support
