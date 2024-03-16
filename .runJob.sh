#!/usr/bin/env bash

# delete previous output from PBS
rm -rf *.rjob_out*

# submit the job to the queue

# Check if an argument is provided
if [ "$#" -eq 0 ]; then
    echo "Error: No argument provided. Please pass an argument."
    exit 1
fi

case $1 in
    "pc") # 
        echo "Running ConvlutionTest.cpp"
		qsub .submission.pbs
        ;;
    "sc") # 
        echo "Running ConvlutionTest.cpp"
		sbatch .submission.slurm >  submission.txt 2> error_cuda.log
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

if [[ ! `cat submission.txt` =~ "Submitted" ]]; then
   echo "Issue submitting..."
   cat submission.txt
   rm -f submission.txt
   exit 1
fi

JOBNUM=`cat submission.txt | awk '{print $4}'`

rm -f submission.txt

# wait for the job to get picked up and start producing output
until [ -f slurm-$JOBNUM.out ]
do 
	sleep 1
done
mv slurm-$JOBNUM.out slurm-$JOBNUM.rjob_out

# open the output file and follow th efile as new output is added
less +F *.rjob_out*