#python a_star.py
#python dijkstras.py --map_loc_filename=../scripts/train_map_configurations.csv

echo "=============================================="
echo "Motion Estimation: Optical Flow"
echo "=============================================="
python learning/scdqn_compare.py --mode=a_star --motion_estimation=of --n_episodes=100
python learning/scdqn_compare.py --mode=dijkstras --motion_estimation=of --n_episodes=100

echo "=============================================="
echo "Motion Estimation: Keypoint Detection"
echo "=============================================="
python learning/scdqn_compare.py --mode=a_star --motion_estimation=kd --n_episodes=100
python learning/scdqn_compare.py --mode=dijkstras --motion_estimation=kd --n_episodes=100

echo "=============================================="
echo "Random Walk"
echo "=============================================="
python learning/scdqn_compare.py --mode=random_walk --motion_estimation=of --n_episodes=100
python learning/scdqn_compare.py --mode=random_walk --motion_estimation=kd --n_episodes=100
