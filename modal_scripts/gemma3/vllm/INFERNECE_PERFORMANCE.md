
================================================================================
COMPREHENSIVE BENCHMARK REPORT
================================================================================

Total benchmark time (parallel): 1680.37s

--------------------------------------------------------------------------------
SETUP TIME COMPARISON
--------------------------------------------------------------------------------
Method                         Setup Time (s)       Winner
--------------------------------------------------------------------------------
HuggingFace Offline            8.85                 ✓ FASTEST
vLLM Offline                   105.60               
vLLM Online                    168.37               

================================================================================
IMAGE SIZE: 224x224
================================================================================

Method                    Throughput (img/s)   Avg Time (ms)        Winner
--------------------------------------------------------------------------------
HuggingFace Offline       13.80                72.46                ✓ FASTEST
vLLM Offline              12.35                80.95                
vLLM Online               2.05                 488.75               

================================================================================
IMAGE SIZE: 512x512
================================================================================

Method                    Throughput (img/s)   Avg Time (ms)        Winner
--------------------------------------------------------------------------------
HuggingFace Offline       13.72                72.91                ✓ FASTEST
vLLM Offline              12.01                83.26                
vLLM Online               2.04                 489.59               

================================================================================
IMAGE SIZE: 1024x1024
================================================================================

Method                    Throughput (img/s)   Avg Time (ms)        Winner
--------------------------------------------------------------------------------
HuggingFace Offline       13.31                75.15                ✓ FASTEST
vLLM Offline              11.28                88.66                
vLLM Online               1.94                 515.70               

================================================================================
OVERALL SUMMARY
================================================================================

Method                         Avg Throughput (img/s)    Overall Winner
--------------------------------------------------------------------------------
HuggingFace Offline            13.61                     ✓ FASTEST OVERALL
vLLM Offline                   11.88                     
vLLM Online                    2.01                      

================================================================================
RECOMMENDATIONS
================================================================================

✓ Fastest Setup: HuggingFace Offline (8.85s)
✓ Fastest Inference: HuggingFace Offline (13.61 img/s)

Use Case Recommendations:
--------------------------------------------------------------------------------
• Quick prototyping / One-off tasks: Choose method with fastest setup
• Large batch processing (>1000 images): Choose method with highest throughput
• Production deployment: Consider vLLM Online for scalability and load balancing
• Offline processing: Consider method with best throughput/memory tradeoff

✓ Benchmark complete!