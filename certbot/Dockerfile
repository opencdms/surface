FROM certbot/certbot

ARG HOST_FQDN
ENV HOST_FQDN ${HOST_FQDN:-localhost.dev}

COPY wait-for-it.sh /wait-for-it.sh
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /wait-for-it.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /var/www/certbot