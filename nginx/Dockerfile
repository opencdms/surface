FROM nginx:1.20.1-alpine

RUN apk add --no-cache bash
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d

CMD ["/wait-for-it.sh", "-t", "30", "api:8000", "--", "nginx", "-g", "daemon off;"]
