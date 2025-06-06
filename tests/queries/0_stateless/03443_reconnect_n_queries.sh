#!/usr/bin/env bash
# Tags: no-fasttest

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

$CLICKHOUSE_BENCHMARK --iterations=10 --reconnect=2 <<< 'SELECT 1' 2>&1 | grep -F 'Queries executed' | tail -n1
