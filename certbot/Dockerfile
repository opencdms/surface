FROM certbot/certbot

ARG HOST_FQDN

ENV HOST_FQDN ${HOST_FQDN:-localhost.dev}

RUN echo $HOST_FQDN
RUN mkdir -p /var/www/certbot