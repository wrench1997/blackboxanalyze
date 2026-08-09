#!/usr/bin/env bash
set -Eeuo pipefail

mysql_home=/usr/local/phpstudy/soft/mysql/mysql-5.7.27
export LD_LIBRARY_PATH="$mysql_home/bin/depends"

# The database lives only in this container's writable layer.  The evaluator
# creates a new --rm, no-volume container for every route episode, so this
# process is never treated as a reset mechanism for a reused target.
"$mysql_home/bin/mysqld_safe" \
  --defaults-file="$mysql_home/my.cnf" \
  --user=www \
  > "$mysql_home/mysql_start.log" 2>&1 &

# Verify the real application credentials before exposing Nginx.  This is a
# local health gate, not a vulnerability oracle and the response is discarded.
for _ in $(seq 1 60); do
  if php -r '$db=@new mysqli("127.0.0.1","root","root","pikachu",3306); exit($db->connect_errno ? 1 : 0);'; then
    break
  fi
  sleep 1
done
if ! php -r '$db=@new mysqli("127.0.0.1","root","root","pikachu",3306); exit($db->connect_errno ? 1 : 0);'; then
  echo "Pikachu MySQL health gate failed" >&2
  exit 1
fi

php-fpm8.0 -D -y /etc/php/8.0/fpm/php-fpm.conf
nginx -t
exec nginx -g 'daemon off;'
