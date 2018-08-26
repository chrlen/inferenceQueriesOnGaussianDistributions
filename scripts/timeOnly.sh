#! /bin/bash
#cd /home/clengert/bachelor/thesis/pythonInference


cd ..

python generateSparseModels.py

python timeAtomicOperations.py models/canonical/ #Done!
python consolidateDataFrames.py atomicOperations.csv models/canonical intel/

python timeInferenceOperations.py models/canonical/
python consolidateDataFrames.py inferenceOperationsSparse.csv models/canonical intel/

python timeInferenceOperationsDense.py models/canonical/
python consolidateDataFrames.py inferenceOperationsDense.csv models/canonical intel/

python timeIndexSet.py
#python timeInversion.py
python timeNPTake.py
#python timeNPTakeSparse.py
#python timeMultiplication.py

