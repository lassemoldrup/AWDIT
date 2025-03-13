#!/bin/sh
set -eu

# Enables RunExec to run inside a Docker container

# Create new sub-cgroups
mkdir -p /sys/fs/cgroup/init /sys/fs/cgroup/benchexec
# Move the init process to that cgroup
echo $$ > /sys/fs/cgroup/init/cgroup.procs

# Enable controllers in subtrees for benchexec to use
for controller in $(cat /sys/fs/cgroup/cgroup.controllers); do
  echo "+$controller" > /sys/fs/cgroup/cgroup.subtree_control
  echo "+$controller" > /sys/fs/cgroup/benchexec/cgroup.subtree_control
done

exec "$@"