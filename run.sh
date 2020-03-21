for ((i = 0; i < 20; i++));
do
    python main.py shapley_identify --seed $i > "./result/shapley_identify/seed-${i}.txt";
done
