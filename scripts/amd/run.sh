clear

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log_$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR


bash scripts/amd/clean.sh
# bash scripts/amd/deps.sh
bash scripts/amd/build.sh
bash scripts/amd/test.sh  2>&1 |tee $LOG_DIR/test.log
# bash scripts/amd/debug.sh
# bash scripts/amd/backtrace.sh 2>&1 |tee $LOG_DIR/backtrace.log

# bash scripts/amd/cache_print.sh  2>&1 |tee $LOG_DIR/cache.log
# bash scripts/amd/post.sh # dont double call