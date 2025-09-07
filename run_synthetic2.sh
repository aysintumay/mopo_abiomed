# python mopo.py --config config/synthetic2/mbpo.yaml --devid 2 
# python mopo.py --config config/synthetic2/mopo.yaml --devid 2 
# python mopo.py --config  config/synthetic2/mopo_shaped.yaml --devid 4 --gamma2 0.3
python mopo.py --config  config/synthetic2/mopo_shaped.yaml --devid 4 --gamma2 0.8 --algo-name mopo_shaped_ws
python mopo.py --config  config/synthetic2/mopo_shaped.yaml --devid 4 --gamma1 0.3 --gamma2 0.0 --algo-name mopo_shaped_acp
python mopo.py --config  config/synthetic2/mopo_shaped.yaml --devid 4 --gamma1 0.5 --gamma2 0.0 --algo-name mopo_shaped_acp
# python mopo.py --config  config/synthetic2/mopo_shaped.yaml --devid 7 --gamma2 0.8

# python mbpo_kde/mopo.py --config config/synthetic2/mbpo_kde.yaml --devid 6 
# python mbpo_kde/mopo.py --config config/synthetic2/mbpo_kde_acp.yaml --devid 6
# python mbpo_kde/mopo.py --config config/synthetic2/mbpo_kde_ws.yaml --devid 6







