# System Architecture & Diagrams

## 1. System Architecture (ASCII)

```mermaid
graph TD
    A[Camera Source / Video File] -->|Frame Stream| B(Preprocessing)
    B -->|Resized Frame| C{YOLOv8 Posture Model}
    C -->|BBox & Class| D[Feature Extractor]
    D -->|Posture, Location, Movement| E[Sequence Buffer]
    E -->|Time Series Data| F{Behaviour Model LSTM}
    F -->|Behaviour & Risk Score| G[Risk Logic Engine]
    G -->|High Risk| H[Alert System]
    G -->|Safe| I[Log / Dashboard]
    H -->|SMS/API| J[User / Ranger]
```

## 2. Dataset Pipeline

```mermaid
graph LR
    A[Raw Images/Video] --> B[Roboflow / LabelImg]
    B -->|Annotate BBox & Class| C[YOLO Dataset]
    C -->|Augmentation| D[Training Set]
    D --> E[YOLOv8 Training]
    
    A --> F[Video Segments]
    F --> G[Extract Features per Frame]
    G -->|CSV Sequence| H[Behaviour Dataset]
    H --> I[LSTM Training]
```

## 3. Training Workflow

```mermaid
flowchart TD
    subgraph Posture_Training
    A[Images] --> B[YOLO Format]
    B --> C[Train YOLOv8]
    C --> D[Best Weights .pt]
    end
    
    subgraph Behaviour_Training
    E[Videos] --> F[Run YOLO Inference]
    F --> G[Extract CSV Features]
    G --> H[Train LSTM/Transformer]
    H --> I[Best Model .pth]
    end
    
    D --> F
```

## 4. Inference & Real-time Alert System

```mermaid
sequenceDiagram
    participant Cam as Camera
    participant Pre as Preprocessor
    participant YOLO as YOLOv8
    participant LSTM as BehaviourModel
    participant Logic as RiskEngine
    participant API as FastAPI
    participant App as AndroidApp

    Cam->>Pre: Send Frame
    Pre->>YOLO: Input Tensor
    YOLO->>Pre: BBox, Class, Conf
    Pre->>LSTM: Update Sequence Buffer
    LSTM->>Logic: Behaviour Label, Risk Score
    Logic->>API: Post Event Data
    
    alt High Risk
        Logic->>API: Trigger Alert
        API->>App: Push Notification
    end
```

## 5. Behaviour Model Architecture

```
Input (Sequence Length T=30)
[Posture_ID, BBox_X, BBox_Y, BBox_W, BBox_H, Speed, Trunk_Angle]
       |
       v
[ LSTM / GRU Layer 1 (Hidden=128) ]
       |
       v
[ LSTM / GRU Layer 2 (Hidden=64) ]
       |
       v
[ Fully Connected Layer ]
       |
       v
[ Softmax Output ]
       |
   (Behaviour Class)
```
