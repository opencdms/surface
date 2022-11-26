export DOLLAR="$"
envsubst < /etc/nginx/conf.d/nginx.conf.template >| /etc/nginx/conf.d/nginx.conf
while :;
do
  sleep 6h & wait $${!};
  /wait-for-it.sh -t 30 api:8000;
  nginx -g "daemon off";
  nginx -s reload;
done