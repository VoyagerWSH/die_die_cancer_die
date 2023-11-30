CUDA_VISIBLE_DEVICES=0,3 python scripts/main.py --monitor_key val_6year_auc --model_name risk_model --dataset_name nlst --project_name part3.1_risk_model
# CUDA_VISIBLE_DEVICES=0,1 python scripts/main.py --model_name risk_model --dataset_name nlst --project_name part3.1_risk_model
# CUDA_VISIBLE_DEVICES=2 python -m debugpy --wait-for-client --listen 1180 scripts/main.py --model_name attn_guided_resnet --dataset_name nlst --project_name part2.1_attn_guided_resnet
# CUDA_VISIBLE_DEVICES=0 python -m debugpy --wait-for-client --listen 1180 scripts/main.py --model_name resnet_3d --dataset_name nlst --project_name part2.1_resnet_3d
# CUDA_VISIBLE_DEVICES=0 python -m debugpy --wait-for-client --listen 1180 scripts/main.py --model_name cnn_3d --dataset_name nlst --project_name part2.1_cnn_3d
# CUDA_VISIBLE_DEVICES=0 python -m debugpy --wait-for-client --listen 1180 scripts/main.py --model_name risk_model --dataset_name nlst --project_name part3.1_risk_model
# CUDA_VISIBLE_DEVICES=0,1 python scripts/main.py --monitor_key val_6year_auc --model_name risk_model --dataset_name nlst --project_name part3.1_risk_model