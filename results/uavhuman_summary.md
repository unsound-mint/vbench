# uavhuman fine-tuning measurements

Hardware: {"gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "gpu_vram_mib": 8188, "driver": "615.71.09", "cuda": "13.0", "torch": "2.14.1+cu130", "cpu": "AMD Ryzen 9 7940HS w/ Radeon 780M Graphics", "os": "Linux 7.2.10-200.fc44.x86_64", "python": "3.12.15"}

## All runs

| lr | model | method | precision | epochs | top1 | macro_f1 | trainable_m | train_min | peak_mem_mib | train_kwh | latency_ms | infer_j_per_clip | val_j | NS_E | NS_M | NS_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | hiera-b | full | bf16 | 3 | 0.68 | 0.02 | 50.904 | 76.56 | 2435.1 | 0.10051 | 39.264 | 3.2811 | 14118.2 | -17.074 | -15.166 | -25.54 |
| default | hiera-b | lora | bf16 | 3 | 51.04 | 51.13 | 0.409 | 66.07 | 1597.3 | 0.08278 | 38.305 | 2.9399 | 13489.0 | 57.991 | 60.308 | 49.983 |
| default | timesformer-b | full | bf16 | 3 | 32.64 | 31.48 | 121.378 | 126.44 | 4722.9 | 0.1675 | 42.932 | 3.8519 | 20886.2 | 49.75 | 51.364 | 40.565 |
| default | timesformer-b | lora | bf16 | 3 | 26.33 | 26.12 | 0.709 | 112.55 | 2895.0 | 0.14695 | 43.653 | 3.9602 | 24179.7 | 45.859 | 48.164 | 37.205 |
| default | videomae-b | adalora | bf16 | 3 | 42.25 | 42.32 | 0.562 | 32.04 | 970.2 | 0.03615 | 15.89 | 1.1276 | 5709.1 | 55.642 | 57.566 | 48.175 |
| default | videomae-b | bitfit | bf16 | 3 | 39.89 | 40.21 | 0.213 | 29.56 | 917.0 | 0.03231 | 15.809 | 1.1007 | 4966.8 | 54.794 | 56.629 | 47.388 |
| default | videomae-b | full | bf16 | 3 | 46.32 | 45.21 | 86.346 | 37.65 | 2127.3 | 0.04225 | 15.819 | 1.0993 | 4934.6 | 57.398 | 58.311 | 49.078 |
| default | videomae-b | full | fp32 | 3 | 47.43 | 46.39 | 86.346 | 109.72 | 2274.7 | 0.16813 | 48.27 | 5.5036 | 19650.3 | 56.309 | 58.65 | 47.916 |
| default | videomae-b | lora | bf16 | 3 | 45.41 | 45.44 | 0.414 | 29.93 | 962.2 | 0.03687 | 15.22 | 1.2393 | 5919.1 | 56.855 | 58.828 | 49.397 |
| default | videomae-b | lora | fp32 | 3 | 45.32 | 45.34 | 0.414 | 97.06 | 1006.6 | 0.13797 | 48.283 | 5.3306 | 19771.4 | 55.512 | 58.744 | 48.004 |
| default | videomae-b | qlora | bf16 | 3 | 44.78 | 44.74 | 0.414 | 34.92 | 619.2 | 0.04322 | 16.757 | 1.2828 | 7271.9 | 56.389 | 59.064 | 49.41 |
| default | videomae-b | qlora | fp16 | 3 | 44.81 | 44.82 | 0.414 | 35.28 | 619.2 | 0.0402 | 17.5 | 1.2532 | 7008.9 | 56.441 | 59.075 | 49.461 |
| default | videoswin-b | full | bf16 | 3 | 46.43 | 45.97 | 87.798 | 123.41 | 4308.3 | 0.14617 | 39.321 | 3.3162 | 18622.2 | 55.997 | 57.586 | 46.911 |
| default | videoswin-b | lora | bf16 | 3 | 32.24 | 32.46 | 0.544 | 103.7 | 2968.6 | 0.12422 | 38.428 | 3.2361 | 18290.0 | 49.68 | 51.654 | 40.999 |

## Learning-rate sensitivity

| lr | model | method | precision | epochs | top1 | macro_f1 | trainable_m | train_min | peak_mem_mib | train_kwh | latency_ms | infer_j_per_clip | val_j | NS_E | NS_M | NS_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | hiera-b | full | bf16 | 3 | 47.82 | 44.24 | 50.904 | 77.56 | 2435.1 | 0.10218 | 39.451 | 3.3434 | 14332.9 | 56.794 | 58.718 | 48.327 |

## Best configuration per model by NS#

| lr | model | method | precision | epochs | top1 | macro_f1 | trainable_m | train_min | peak_mem_mib | train_kwh | latency_ms | infer_j_per_clip | val_j | NS_E | NS_M | NS_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | hiera-b | lora | bf16 | 3 | 51.04 | 51.13 | 0.409 | 66.07 | 1597.3 | 0.08278 | 38.305 | 2.9399 | 13489.0 | 57.991 | 60.308 | 49.983 |
| default | timesformer-b | full | bf16 | 3 | 32.64 | 31.48 | 121.378 | 126.44 | 4722.9 | 0.1675 | 42.932 | 3.8519 | 20886.2 | 49.75 | 51.364 | 40.565 |
| default | videomae-b | qlora | fp16 | 3 | 44.81 | 44.82 | 0.414 | 35.28 | 619.2 | 0.0402 | 17.5 | 1.2532 | 7008.9 | 56.441 | 59.075 | 49.461 |
| default | videoswin-b | full | bf16 | 3 | 46.43 | 45.97 | 87.798 | 123.41 | 4308.3 | 0.14617 | 39.321 | 3.3162 | 18622.2 | 55.997 | 57.586 | 46.911 |

## Best configuration per model by NS_M

| lr | model | method | precision | epochs | top1 | macro_f1 | trainable_m | train_min | peak_mem_mib | train_kwh | latency_ms | infer_j_per_clip | val_j | NS_E | NS_M | NS_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | hiera-b | lora | bf16 | 3 | 51.04 | 51.13 | 0.409 | 66.07 | 1597.3 | 0.08278 | 38.305 | 2.9399 | 13489.0 | 57.991 | 60.308 | 49.983 |
| default | timesformer-b | full | bf16 | 3 | 32.64 | 31.48 | 121.378 | 126.44 | 4722.9 | 0.1675 | 42.932 | 3.8519 | 20886.2 | 49.75 | 51.364 | 40.565 |
| default | videomae-b | qlora | fp16 | 3 | 44.81 | 44.82 | 0.414 | 35.28 | 619.2 | 0.0402 | 17.5 | 1.2532 | 7008.9 | 56.441 | 59.075 | 49.461 |
| default | videoswin-b | full | bf16 | 3 | 46.43 | 45.97 | 87.798 | 123.41 | 4308.3 | 0.14617 | 39.321 | 3.3162 | 18622.2 | 55.997 | 57.586 | 46.911 |

## Best configuration per model by NS_E

| lr | model | method | precision | epochs | top1 | macro_f1 | trainable_m | train_min | peak_mem_mib | train_kwh | latency_ms | infer_j_per_clip | val_j | NS_E | NS_M | NS_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| default | hiera-b | lora | bf16 | 3 | 51.04 | 51.13 | 0.409 | 66.07 | 1597.3 | 0.08278 | 38.305 | 2.9399 | 13489.0 | 57.991 | 60.308 | 49.983 |
| default | timesformer-b | full | bf16 | 3 | 32.64 | 31.48 | 121.378 | 126.44 | 4722.9 | 0.1675 | 42.932 | 3.8519 | 20886.2 | 49.75 | 51.364 | 40.565 |
| default | videomae-b | full | bf16 | 3 | 46.32 | 45.21 | 86.346 | 37.65 | 2127.3 | 0.04225 | 15.819 | 1.0993 | 4934.6 | 57.398 | 58.311 | 49.078 |
| default | videoswin-b | full | bf16 | 3 | 46.43 | 45.97 | 87.798 | 123.41 | 4308.3 | 0.14617 | 39.321 | 3.3162 | 18622.2 | 55.997 | 57.586 | 46.911 |
