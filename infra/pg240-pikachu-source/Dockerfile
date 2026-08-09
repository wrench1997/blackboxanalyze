FROM sift/pikachu-fixed:pg214

LABEL org.opencontainers.image.title="Pikachu source replay variant"
LABEL org.opencontainers.image.description="Pinned upstream Pikachu application source on the locally repaired PHP/MySQL runtime"
LABEL org.opencontainers.image.source.commit="5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc"

# The build context is the pinned upstream repository checkout.  Only the
# application tree is replaced; the already-attested local runtime remains
# unchanged so the comparison isolates application-source behavior.
COPY . /app/www/

# The upstream checkout documents an empty root password; the fixed runtime's
# disposable MySQL uses root/root.  This is a local connection setting, not a
# payload or evaluator label.
RUN sed -i "s/define('DBPW', '');/define('DBPW', 'root');/" /app/www/inc/config.inc.php \
    && find /app/www -type f -name '*.php' -print0 | xargs -0 -n1 php -l >/dev/null
