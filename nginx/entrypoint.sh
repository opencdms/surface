/init-letsencrypt.sh
export DOLLAR="$"
envsubst < /etc/nginx/conf.d/nginx.conf.template >| /etc/nginx/conf.d/nginx.conf
/wait-for-it.sh -t 30 api:8000;
nginx -g 'daemon off;';
while :;
do
  sleep 6h;
  nginx -s reload;
done