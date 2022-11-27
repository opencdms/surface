FROM nginx:1.20.1-alpine

RUN apk add --no-cache bash
COPY entrypoint.sh /entrypoint.sh
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh
RUN chmod +x /entrypoint.sh

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/nginx.conf.template

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
