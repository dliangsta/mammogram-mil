# python3 train.py \
#   --model_name "mil" \
#   --mil_type "stack" \
#   --image_dir "../data/stack_case_images_large" \
#   --positive_error_rate_multiplier 2. \
#   --learning_rate 1e-5 \
#   --regularization_loss_weight 1e-7 \
#   --normalize_input \
#   --image_height 1024 \
#   --image_width 1024 \
#   --large \
#   $1
# python3 train.py \
#   --model_name "mil" \
#   --mil_type "vote" \
#   --vote_type "nn" \
#   --positive_error_rate_multiplier 1. \
#   --learning_rate 1e-7 \
#   --regularization_loss_weight 1e-7 \
#   --normalize_input \
#   --image_height 598 \
#   --image_width 598 \
#   --image_channels 4 \
#   --augment
#   $1
python3 train.py \
  --model_name "baseline" \
  --positive_error_rate_multiplier 1. \
  --learning_rate 1e-7 \
  --regularization_loss_weight 1e-7 \
  --normalize_input \
  --image_height 299 \
  --image_width 299 \
  --image_channels 3 \
  --augment
  $1
# python3 train.py \
#   --model_name "mil" \
#   --mil_type "stack" \
#   --image_dir "../data/stack_case_images" \
#   --learning_rate 1e-7 \
#   --positive_error_rate_multiplier 1. \
#   --regularization_loss_weight 1e-7 \
#   --normalize_input \
#   --augment \
#   --image_height 299 \
#   --image_width 299 \
#   $1
# python3 train.py --model_name "baseline" --normalize --augment $1
# python3 train.py --model_name "transfer" --normalize --augment $1