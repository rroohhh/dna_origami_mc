for t in $(seq -5 0.5 -1); do
    for i in $(seq 1 10); do
        real_t=$(awk "BEGIN{print 10**$t}")
        echo "$real_t $i"
        python3 newVersionTest.py $real_t
    done
done
