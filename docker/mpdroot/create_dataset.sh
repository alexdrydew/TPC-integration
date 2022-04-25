git clone https://github.com/SiLiKhon/TPC-FastSim
mkdir -p $OUTPUT_DIR/$DATASET_DIR/raw
cp -r TPC-FastSim/data/data_v4/raw/* $OUTPUT_DIR/$DATASET_DIR/raw
chown -R 777 $OUTPUT_DIR/$DATASET_DIR
