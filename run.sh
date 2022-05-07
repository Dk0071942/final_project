echo "=======MUTAG======="
echo "=======GIN======="
python main.py --dataset='MUTAG' --GNN_model='GIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='GIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='GIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='GIN' --hidden_units=16
echo "=======DropGIN======="
python main.py --dataset='MUTAG' --GNN_model='DropGIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='DropGIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='DropGIN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='DropGIN' --hidden_units=16
echo "=======1_2_3_GNN======="
python main.py --dataset='MUTAG' --GNN_model='1_2_3_GNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='1_2_3_GNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='1_2_3_GNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='1_2_3_GNN' --hidden_units=16
echo "=======TDGNN======="
python main.py --dataset='MUTAG' --GNN_model='TDGNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='TDGNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='TDGNN' --hidden_units=16
python main.py --dataset='MUTAG' --GNN_model='TDGNN' --hidden_units=16
# python main.py --dataset='PTC_MR' --GNN_model='GIN' --hidden_units=16
# python main.py --dataset='PTC_MR' --GNN_model='DropGIN' --hidden_units=16
# python main.py --dataset='PTC_MR' --GNN_model='1_2_3_GNN' --hidden_units=16
# python main.py --dataset='PTC_MR' --GNN_model='TDGNN' --hidden_units=16
# python main.py --dataset='PROTEINS' --GNN_model='GIN' --hidden_units=16
# python main.py --dataset='PROTEINS' --GNN_model='DropGIN' --hidden_units=16
# python main.py --dataset='PROTEINS' --GNN_model='1_2_3_GNN' --hidden_units=16
# python main.py --dataset='PROTEINS' --GNN_model='TDGNN' --hidden_units=16
# python main.py --dataset='IMDB-BINARY' --GNN_model='GIN' --hidden_units=64
# python main.py --dataset='IMDB-BINARY' --GNN_model='DropGIN' --hidden_units=64
# python main.py --dataset='IMDB-BINARY' --GNN_model='1_2_3_GNN' --hidden_units=64
# python main.py --dataset='IMDB-BINARY' --GNN_model='TDGNN' --hidden_units=64
# python main.py --dataset='IMDB-MULTI' --GNN_model='GIN' --hidden_units=64
# python main.py --dataset='IMDB-MULTI' --GNN_model='DropGIN' --hidden_units=64
# python main.py --dataset='IMDB-MULTI' --GNN_model='1_2_3_GNN' --hidden_units=64
# python main.py --dataset='IMDB-MULTI' --GNN_model='TDGNN' --hidden_units=64
