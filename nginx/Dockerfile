FROM nginx:1.20.1-alpine

RUN apk add --no-cache bash certbot certbot-nginx

COPY entrypoint.sh /entrypoint.sh
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh
RUN chmod +x /entrypoint.sh

RUN rm /etc/nginx/conf.d/default.conf
RUN mkdir -p /etc/nginx-conf-template/
COPY nginx.conf.local /etc/nginx-conf-template/nginx.conf.local.template
COPY nginx.conf.http /etc/nginx-conf-template/nginx.conf.http.template
COPY nginx.conf.https /etc/nginx-conf-template/nginx.conf.https.template

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
