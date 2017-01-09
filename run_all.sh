counter = 0
for rom in $(ls ../roms/)
do
    let counter=counter+1
    echo "playing $rom ..."
    echo $counter
    let ctx=$((counter / 17 +1)) 
    echo $ctx
    python run_nature.py -r $rom --ctx $ctx &
done
