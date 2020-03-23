for ((i = 25; i < 30; i++));
do
    #python main.py influence_identify --seed $i > "./result/influence_identify/seed-${i}.txt";
    python main.py shapley_identify --seed $i > "./result/shapley_identify/seed-${i}.txt";
done
