# model_name_train_map#




python learning/qnn.py --gamma=0.95 --map_num=1 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_of_map1.csv --video_df_filename=./scripts/map_config/optical_flow_train_map1_config.csv --mode=train --optimizer=SGD --motion_estimation=of
python learning/qnn.py --gamma=0.95 --map_num=2 --alpha=0.0001 --n_episodes=3000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_of_map2.csv --video_df_filename=./scripts/map_config/optical_flow_train_map2_config.csv --mode=train --optimizer=SGD --motion_estimation=of
python learning/qnn.py --gamma=0.95 --map_num=3 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_of_map3.csv --video_df_filename=./scripts/map_config/optical_flow_train_map3_config.csv --mode=train --optimizer=SGD --motion_estimation=of
python learning/qnn.py --gamma=0.95 --map_num=4 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_of_map4.csv --video_df_filename=./scripts/map_config/optical_flow_train_map4_config.csv --mode=train --optimizer=SGD --motion_estimation=of


python learning/qnn.py --gamma=0.95 --map_num=1 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_kd_map1.csv --video_df_filename=./scripts/map_config/keypoint_train_map1_config.csv --mode=train --optimizer=SGD --motion_estimation=kd
python learning/qnn.py --gamma=0.95 --map_num=2 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_kd_map2.csv --video_df_filename=./scripts/map_config/keypoint_train_map2_config.csv --mode=train --optimizer=SGD --motion_estimation=kd
python learning/qnn.py --gamma=0.95 --map_num=3 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_kd_map3.csv --video_df_filename=./scripts/map_config/keypoint_train_map3_config.csv --mode=train --optimizer=SGD --motion_estimation=kd
python learning/qnn.py --gamma=0.95 --map_num=4 --alpha=0.0001 --n_episodes=10000 --epsilon=0.1 --rendering=0 --save_map=./networks/train_kd_map4.csv --video_df_filename=./scripts/map_config/keypoint_train_map4_config.csv --mode=train --optimizer=SGD --motion_estimation=kd