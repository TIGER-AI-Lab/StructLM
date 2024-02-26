# make the port a variable

REMOTE_PORT=30019
RUN_NAME=$1
#ex. "non_upsampled_e2"
DATASET_NAME=$2
#ex. "wikisql"
REMOTE="huangwenhao@8.130.143.52:/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/"
# use identity file at /home/alex/v3-score/azhx
# create an output directory at output/${RUN_NAME}_${DATASET_NAME}

mkdir -p output/${RUN_NAME}_${DATASET_NAME}

# if the file summary.json already exists, skip this dataset
if [ -f output/${RUN_NAME}_${DATASET_NAME}/summary.json ]; then
    echo "output/${RUN_NAME}_${DATASET_NAME}/summary.json already exists, skipping..."
    exit 0
fi

echo "Pulling ${DATASET_NAME} from ${REMOTE} to output/${RUN_NAME}_${DATASET_NAME}/"

# copy the predictions
scp -oStrictHostKeyChecking=no -i /home/alex/v3-score/azhx -P $REMOTE_PORT \
    ${REMOTE}/output/${RUN_NAME}_${DATASET_NAME}/predictions_predict.json\
    ./output/${RUN_NAME}_${DATASET_NAME}/

# copy the config file
scp -oStrictHostKeyChecking=no -i /home/alex/v3-score/azhx -P $REMOTE_PORT \
    ${REMOTE}/configure/Salesforce/${RUN_NAME}_${DATASET_NAME}.cfg\
    ./configure/Salesforce/

# TODO: run python script that does the eval for that specific dataset
python eval_json.py --dataset_name ${DATASET_NAME} --run_name ${RUN_NAME}
