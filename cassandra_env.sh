# Memory settings for Cassandra
export MAX_HEAP_SIZE="2G"
export HEAP_NEWSIZE="512M"

# JVM Options
export JVM_OPTS="$JVM_OPTS -XX:+UseG1GC"
export JVM_OPTS="$JVM_OPTS -XX:G1RSetUpdatingPauseTimePercent=5"
export JVM_OPTS="$JVM_OPTS -XX:MaxGCPauseMillis=500"
export JVM_OPTS="$JVM_OPTS -XX:ParallelGCThreads=8"
export JVM_OPTS="$JVM_OPTS -XX:ConcGCThreads=2"