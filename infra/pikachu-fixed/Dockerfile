FROM tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# The pinned image carries a PHP 5.5 binary whose private libraries are no
# longer present in Ubuntu 21.10.  Keep the application and database image
# immutable, but install the archive-pinned PHP 8.0 FPM + mysqli runtime so
# the existing Nginx/Pikachu routes can actually execute.
RUN set -eux; \
    sed -i 's@archive.ubuntu.com/ubuntu@old-releases.ubuntu.com/ubuntu@g; s@security.ubuntu.com/ubuntu@old-releases.ubuntu.com/ubuntu@g' /etc/apt/sources.list; \
    printf '%s\n' \
      'Acquire::AllowInsecureRepositories "true";' \
      'APT::Get::AllowUnauthenticated "true";' \
      > /etc/apt/apt.conf.d/99-pikachu-archive; \
    apt-get update -qq; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      php8.0-fpm=8.0.8-1ubuntu0.4 \
      php8.0-mysql=8.0.8-1ubuntu0.4; \
    # PHP 8 removed the old curly-brace string-offset syntax used by the
    # bundled helper; this mechanical compatibility edit does not alter any
    # route logic or the intentionally vulnerable query surfaces. \
    sed -i 's/\$string{\$pos}/\$string[\$pos]/' /app/www/inc/function.php; \
    find /app/www -type f -name '*.php' -print0 | xargs -0 -n1 php -l >/dev/null; \
    sed -i 's@^listen = .*@listen = 127.0.0.1:5538@' /etc/php/8.0/fpm/pool.d/www.conf; \
    mkdir -p /run/php; \
    php-fpm8.0 -tt; \
    nginx -t; \
    rm -rf /var/lib/apt/lists/*

COPY run-fixed.sh /app/run-fixed.sh
RUN chmod 0755 /app/run-fixed.sh

EXPOSE 8090 3306
ENTRYPOINT ["/app/run-fixed.sh"]
