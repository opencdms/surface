FROM certbot/certbot

ENV HOST_FQDN ${HOST_FQDN:-localhost.dev}

RUN echo $HOST_FQDN
RUN mkdir -p /var/www/certbot
RUN certbot certonly --webroot --webroot-path /var/www/certbot/ -d ${HOST_FQDN} --non-interactive --agree-tos -m admin@${HOST_FQDN}